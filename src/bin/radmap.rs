use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use array_lib::{io_nifti, io_nrrd, ArrayDim};
use array_lib::io_nifti::{write_nifti_with_header, NiftiHeader};
use array_lib::io_nrrd::{write_nrrd, Encoding, NRRD};
use indicatif::{ProgressBar, ProgressStyle};
use clap::Parser;
use glcm::core::GLCMFeature;
use glcm::run_glcm_map;
use glcm::ui::MapOpts;

#[derive(Parser, Debug)]
pub struct Args {
    /// input volume to generate feature maps from
    input_vol: PathBuf,

    /// output directory to write results
    output_dir: PathBuf,

    /// optional mask to limit number of voxels to accelerate calculations
    mask: Option<PathBuf>,

    /// number of bins for the GLCM, 32 bins is default
    n_bins: Option<usize>,

    /// determines the shell of voxels considered to be neighbors. Default is 1
    kernel_radius: Option<i32>,

    #[clap(short, long)]
    /// include all features. If `omit` is specified for some features, they will be
    /// removed from the collection
    all_features: bool,

    #[clap(short, long)]
    /// supply a single feature to include (multiple can be included with additional -f flags)
    feature: Vec<String>,

    #[clap(long)]
    /// supply a single feature to omit from calculations (multiple can be omitted with additional --omit flags)
    omit: Vec<String>,

    #[clap(long)]
    /// limit the number of parallel worker threads. This is the number of logical CPU cores
    max_threads:Option<usize>,

    #[clap(short,long)]
    /// print a progress bar
    progress: bool

}

fn main() {

    let args = Args::parse();

    let mut opts = MapOpts::default();
    opts.n_bins = args.n_bins.unwrap_or(32);
    opts.kernel_radius = args.kernel_radius.map(|r| r.abs() as usize).unwrap_or(1);
    opts.max_threads = args.max_threads;

    if !args.output_dir.is_dir() {
        panic!("Output directory {} does not exist",args.output_dir.display());
    }

    if !args.input_vol.is_file() {
        panic!("Input volume {} file does not exist",args.input_vol.display());
    }

    let input_stem = args.input_vol.file_stem().unwrap().to_str().unwrap();

    if !args.all_features {
        opts.features.clear();
        for f in args.feature {
            let feature = GLCMFeature::from_str(&f.to_lowercase()).expect(&format!("Invalid GLCM feature: {}", f));
            opts.features.insert(feature,feature.to_string().to_lowercase());
        }
    }

    for to_omit in args.omit {
        let feature = GLCMFeature::from_str(&to_omit.to_lowercase()).expect(&format!("Invalid GLCM feature: {}", to_omit));
        opts.features.remove(&feature);
    }

    if opts.features.is_empty() {
        panic!("No features specified!");
    }

    let (vol, dims, header) = read_volume(&args.input_vol);
    let mask = if let Some(mask) = &args.mask {
        let (mask_vol, mask_dims, ..) = read_volume(mask);
        assert_eq!(dims.shape_ns(), mask_dims.shape_ns(), "input volume and mask must have the same shape");
        Some(mask_vol)
    }else {
        None
    };

    let vox_to_process = dims.numel() as u64;

    let masked_voxels = mask.as_ref().map(|mask| mask.iter().filter(|x| **x != 0.).count()).unwrap_or(dims.numel());
    let n_features = opts.features.len();
    println!("launching GLCM mapper for {n_features} features over {masked_voxels} voxels ...");

    let progress = Arc::new(AtomicUsize::new(0));
    let t_progress = progress.clone();
    let t_dims = dims.clone();
    let t_opts = opts.clone();
    let now = Instant::now();
    let h = thread::spawn(move||{
        run_glcm_map(t_opts, vol, mask, t_dims, t_progress)
    });

    if args.progress {
        let pb = ProgressBar::new(vox_to_process);
        pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("##-"));
        while pb.position() < vox_to_process {
            let val = progress.load(Ordering::Relaxed) as u64;
            pb.set_position(val);
            thread::sleep(Duration::from_millis(15));
        }
        pb.finish_with_message("all voxels mapped successfully");
    }

    let (results,..) = h.join().expect("Failed to join thread");

    let duration = now.elapsed();
    println!("{} voxels processed in {} minutes", masked_voxels, duration.as_secs_f64() / 60.);

    println!("writing outputs to {}",args.output_dir.display());
    let vol_stride = dims.numel();
    for (&f, alias) in opts.features.iter() {
        let i = f as usize;
        let vol = &results[i * vol_stride..(i + 1) * vol_stride];
        let path = args.output_dir.join(format!(
            "{}{}{}",
            input_stem,
            "_",
            alias.to_lowercase().replace(" ", "_")
        ));
        match &header {
            Header::Nrrd(nhdr) => write_nrrd(path, vol, dims, Some(nhdr), false, Encoding::raw),
            Header::Nifti(nii) => write_nifti_with_header(path, vol, dims, nii),
        };
    }
}

enum Header {
    Nrrd(Box<NRRD>),
    Nifti(Box<NiftiHeader>),
}

fn read_volume(path:impl AsRef<Path>) -> (Vec<f64>, ArrayDim, Header) {
    let vol_path = path.as_ref().to_path_buf();

    let f_ext = vol_path.extension().expect("file has no extension").to_str().unwrap();

    if f_ext == "nii" || f_ext == "nii.gz" {
        let (data, dims, header) = io_nifti::read_nifti::<f64>(vol_path);
        (data, dims, Header::Nifti(Box::new(header)))
    } else if f_ext == "nhdr" || f_ext == "nrrd" {
        let (data, dims, header) = io_nrrd::read_nrrd(vol_path);
        (data, dims, Header::Nrrd(Box::new(header)))
    } else {
        panic!("Only nrrds and niftis are supported. Sorry :)");
    }
}

fn write_volume(dir:impl AsRef<Path>, vol:&[f32], vol_dims:ArrayDim, header:&Header) {

    match &header {
        Header::Nrrd(nhdr) => {
            write_nrrd(dir, vol, vol_dims, Some(nhdr), false, Encoding::raw)
        }
        Header::Nifti(nii) => write_nifti_with_header(dir, vol, vol_dims, nii),
    };

}