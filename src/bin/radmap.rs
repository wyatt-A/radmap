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
use strum::IntoEnumIterator;
use rayon::prelude::*;
use rayon::current_num_threads;

#[derive(Parser, Debug)]
pub struct Args {
    /// input volume to generate feature maps from
    #[arg(required_unless_present = "list_features")]
    input_vol: Option<PathBuf>,

    /// output directory to write results
    #[arg(required_unless_present = "list_features")]
    output_dir: Option<PathBuf>,

    /// optional mask to limit number of voxels to accelerate calculations
    #[clap(short, long)]
    mask: Option<PathBuf>,

    /// list all glcm features for reference
    #[clap(short, long)]
    list_features: bool,

    /// number of bins for the GLCM, 32 bins is default
    #[clap(short, long)]
    n_bins: Option<usize>,

    /// determines the shell of voxels considered to be neighbors. Default is 1
    #[clap(short, long)]
    kernel_radius: Option<i32>,

    /// include all features. If `omit` is specified for some features, they will be
    /// removed from the collection
    #[clap(short, long)]
    all_features: bool,

    /// supply a single feature to include (multiple can be included with additional -f flags)
    #[clap(short, long)]
    feature: Vec<String>,

    /// supply a single feature to omit from calculations (multiple can be omitted with additional --omit flags)
    #[clap(long)]
    omit: Vec<String>,

    /// limit the number of parallel worker threads. This is the number of logical CPU cores
    #[clap(long)]
    max_threads:Option<usize>,

    /// print the progress bar. To disable, pass --progress false
    #[clap(long, default_value="true")]
    progress: bool

}

fn main() {

    let args = Args::parse();

    if args.list_features {
        for f in GLCMFeature::iter() {
            println!("{}",f.to_string().to_lowercase());
        }
        return
    }

    let mut opts = MapOpts {
        n_bins: args.n_bins.unwrap_or(32),
        kernel_radius: args.kernel_radius.map(|r| r.unsigned_abs() as usize).unwrap_or(1),
        max_threads: args.max_threads,
        ..Default::default()
    };

    println!("num bins: {}",opts.n_bins);
    println!("kernel radius: {}",opts.kernel_radius);
    if let Some(threads) = opts.max_threads {
        println!("limiting max logical cores to {}",threads);
    }else {
        let logical_cores = current_num_threads();
        println!("using all {logical_cores} logical cores for processing");
    }

    let output_dir = args.output_dir.as_ref().unwrap();
    let input_vol = args.input_vol.as_ref().unwrap();

    if !output_dir.is_dir() {
        panic!("Output directory {} does not exist",output_dir.display());
    }

    if !input_vol.is_file() {
        panic!("Input volume {} file does not exist",input_vol.display());
    }

    let input_stem = input_vol.file_stem().unwrap().to_str().unwrap();

    if !args.all_features {
        opts.features.clear();
        for f in args.feature {
            let feature = GLCMFeature::from_str(&f.to_lowercase()).unwrap_or_else(|_| panic!("Invalid GLCM feature: {}", f));
            opts.features.insert(feature,feature.to_string().to_lowercase());
        }
    }

    for to_omit in args.omit {
        let feature = GLCMFeature::from_str(&to_omit.to_lowercase()).unwrap_or_else(|_| panic!("Invalid GLCM feature: {}", to_omit));
        opts.features.remove(&feature);
    }

    if opts.features.is_empty() {
        panic!("No features specified!");
    }

    println!("loading volume ...");
    let (vol, dims, header) = read_volume(input_vol);
    let mask = if let Some(mask) = &args.mask {
        println!("loading mask ...");
        let (mask_vol, mask_dims, ..) = read_volume(mask);
        assert_eq!(dims.shape_ns(), mask_dims.shape_ns(), "input volume and mask must have the same shape");
        Some(mask_vol)
    }else {
        None
    };

    let vox_to_process = dims.numel() as u64;

    let masked_voxels = mask.as_ref().map(|mask| mask.par_iter().filter(|x| **x != 0.).count()).unwrap_or(dims.numel());
    let n_features = opts.features.len();
    println!("launching GLCM mapper for {n_features} feature(s) over {masked_voxels} voxels ...");

    let progress = Arc::new(AtomicUsize::new(0));
    let t_progress = progress.clone();
    let t_dims = dims;
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
            thread::sleep(Duration::from_millis(100));
        }
        pb.finish_with_message("all voxels mapped successfully");
        print!("\n");
    }

    let (results,..) = h.join().expect("Failed to join thread");

    let duration = now.elapsed();
    println!("{} voxels processed in {:.03} minutes", masked_voxels, duration.as_secs_f64() / 60.);

    println!("writing outputs to {}",output_dir.display());
    let vol_stride = dims.numel();
    for (&f, alias) in opts.features.iter() {
        let i = f as usize;
        let vol = &results[i * vol_stride..(i + 1) * vol_stride];
        let path = output_dir.join(format!(
            "{}{}{}",
            input_stem,
            "_",
            alias.to_lowercase().replace(" ", "_")
        ));
        write_volume(path, vol, dims, &header);
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

fn write_volume(path:impl AsRef<Path>, vol:&[f32], vol_dims:ArrayDim, header:&Header) {

    match &header {
        Header::Nrrd(nhdr) => {
            write_nrrd(path, vol, vol_dims, Some(nhdr), false, Encoding::raw)
        }
        Header::Nifti(nii) => write_nifti_with_header(path, vol, vol_dims, nii),
    };

}