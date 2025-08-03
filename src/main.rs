use std::collections::HashMap;
use strum::IntoEnumIterator;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use eframe::{egui, Frame, NativeOptions};
use eframe;
use eframe::egui::{vec2, Color32, Context, IconData, ProgressBar, RichText, Ui};
use egui_file_dialog::FileDialog;
use glcm::run_glcm_map;
use glcm::ui::MapOpts;
use glcm::glcm::GLCMFeature;
use array_lib::{io_nifti, io_nrrd, ArrayDim};
use array_lib::io_nifti::{write_nifti_with_header, NiftiHeader};
use array_lib::io_nrrd::{write_nrrd, Encoding, NRRD};

const ICON_BYTES: &[u8] = include_bytes!("../assets/icon.png");

fn main() {

    let icon = {
        let img = image::load_from_memory(ICON_BYTES)
            .expect("Failed to decode embedded PNG")
            .into_rgba8();
        let (width,height) = img.dimensions();
        IconData { width, height, rgba: img.into_raw() }
    };

    let mut native_options = NativeOptions::default();
    native_options.vsync = true;
    native_options.viewport.icon = Some(Arc::new(icon));
    native_options.viewport.inner_size = Some(vec2( 1000.0,600.0));

    eframe::run_native(
        "RadMap",
        native_options,
        Box::new(|_cc| Ok(Box::new(GUI::default()))),
    ).unwrap();
}


pub struct GUI {
    data_loader: InputSelector,
    output_selector: OutputSelector,
    opts_selector: MapOptSelector,
    feature_selector: FeatureSelector,
    glcm_launcher: GLCMLauncher,
    progress: Progress,
    map_opts: MapOpts,
}

impl Default for GUI {
    fn default() -> Self {
        GUI {
            data_loader: InputSelector::default(),
            output_selector: Default::default(),
            opts_selector: MapOptSelector::default(),
            feature_selector: Default::default(),
            glcm_launcher: Default::default(),
            progress: Default::default(),
            map_opts: MapOpts::default(),
        }
    }
}

impl eframe::App for GUI {
    fn update(&mut self, ctx: &Context, frame: &mut Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {

            ui.columns(2, |columns| {


                columns[0].vertical(|ui| {
                    update_map_options(&mut self.opts_selector, ctx, ui);
                    update_feature_selector(&mut self.feature_selector, ui);
                });

                columns[1].vertical(|ui| {
                    update_data_loader(&mut self.data_loader, ctx,  ui);
                    update_output_selector(&mut self.output_selector,&self.data_loader,&self.feature_selector,&mut self.glcm_launcher, ctx,  ui);

                    update_glcm_launcher(
                        &mut self.map_opts, &mut self.progress, &mut self.glcm_launcher,&self.opts_selector,
                        &self.feature_selector, &self.data_loader,&self.output_selector,ui
                    );

                    update_progress(&mut self.progress, ui);

                });

            });

        });
        ctx.request_repaint();
    }
}

pub fn update_options(opts:&mut MapOpts, map_opts:&MapOptSelector, features:&FeatureSelector) {

    opts.kernel_radius = map_opts.kernel_radius;
    opts.features = features.selected_features.clone();
    opts.n_bins = map_opts.num_bins;
    opts.separator = None;

}

// /****************************
// ******* DATA EXPORT ********
// ****************************/
//
// pub struct Export {
//
//     /// save data as single precision 32-bit float
//     save_float: bool,
//
// }
//
// impl Default for Export {
//     fn default() -> Self {
//         Export {
//             save_float: true,
//         }
//     }
// }
//
// pub fn update_export(export:&mut Export, launcher:&mut GLCMLauncher,  ui:&mut Ui) {
//
//
// }


/****************************
******* GLCM LAUNCHER *******
****************************/
pub struct GLCMLauncher {
    result: Option<(Vec<f32>,ArrayDim)>,
    ref_header: Option<Header>,
    handle:Option<JoinHandle<(Vec<f32>,ArrayDim)>>,
    is_running: bool,
    succeeded: bool,
}

impl Default for GLCMLauncher {
    fn default() -> Self {
        GLCMLauncher {
            result: None,
            ref_header: None,
            handle: None,
            is_running: false,
            succeeded: false,
        }
    }
}

pub enum Header {
    Nrrd(NRRD),
    Nifti(NiftiHeader)
}

pub fn update_glcm_launcher(map_opts:&mut MapOpts, progress:&mut Progress, launcher: &mut GLCMLauncher, opts_selector:&MapOptSelector, features:&FeatureSelector, data_selector:&InputSelector, output_selector: &OutputSelector, ui:&mut Ui) {

    // check that files have been selected
    if data_selector.volume_path.is_some() && output_selector.output_dir.is_some() {
        update_options(map_opts, opts_selector, features);

        if ui.button("LAUNCH").clicked() {
            let vol_path = data_selector.volume_path.as_ref().unwrap().clone();
            let vol_handle = std::thread::spawn(move || {
                if vol_path.extension().unwrap() == "nii" || vol_path.extension().unwrap() == "nii.gz" {
                    let (data, dims, header) = io_nifti::read_nifti::<f64>(vol_path);
                    (data, dims, Header::Nifti(header))
                } else {
                    let (data, dims, header) = io_nrrd::read_nrrd(vol_path);
                    (data, dims, Header::Nrrd(header))
                }
            });

            let mask_handle = if let Some(mask_path) = &data_selector.mask_path {
                let mp = mask_path.clone();
                let h = std::thread::spawn(move || {
                    if mp.extension().unwrap() == "nii" || mp.extension().unwrap() == "nii.gz" {
                        let (data, dims, header) = io_nifti::read_nifti::<f64>(mp);
                        (data, dims, Header::Nifti(header))
                    } else {
                        let (data, dims, header) = io_nrrd::read_nrrd(mp);
                        (data, dims, Header::Nrrd(header))
                    }
                });
                Some(h)
            } else {
                None
            };

            let (vol, vol_dims, vol_header) = vol_handle.join().expect("failed to retrieve volume from loader thread");
            let mask = mask_handle.map(|h| h.join().expect("failed to retrieve mask from loader thread"));

            progress.total_vox_to_compute = Some(vol_dims.numel());

            let t_map_opts = map_opts.clone();
            progress.progress = Arc::new(AtomicUsize::new(0));
            let t_progress = progress.progress.clone();
            let glcm_calc_handle = std::thread::spawn(move || {
                // check that the mask and volume have compatible shapes
                let mask = mask.map(|(mask_data, mask_dims, _)| {
                    assert_eq!(mask_dims.shape_ns(), vol_dims.shape_ns(), "mask and volume have different shapes");
                    mask_data
                });
                run_glcm_map(t_map_opts, vol, mask, vol_dims, t_progress)
            });

            launcher.is_running = true;
            launcher.succeeded = false;
            launcher.ref_header = Some(vol_header);
            launcher.handle = Some(glcm_calc_handle);
        }
    }

    if let Some(h) = launcher.handle.take() {
        if h.is_finished() {
            let result = h.join().expect("failed to retrieve handle from calc thread");
            launcher.result = Some(result);
            launcher.is_running = false;
            launcher.succeeded = true;
        }else {
            launcher.handle = Some(h);
        }
    }

    if launcher.is_running {
        ui.label("running ...");
    }

    if launcher.succeeded {
        ui.label("feature calculations succeeded");
    }

}


/****************************
********** PROGRESS *********
****************************/
pub struct Progress {
    progress: Arc<AtomicUsize>,
    total_vox_to_compute: Option<usize>,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
            total_vox_to_compute: None,
        }
    }
}

pub fn update_progress(progress:&mut Progress,ui:&mut Ui) {
    if let Some(total_vox) = progress.total_vox_to_compute {
        let state = progress.progress.load(Ordering::Relaxed);
        let progress = state as f64 / total_vox as f64;
        ui.add(ProgressBar::new(progress as f32).show_percentage());
    }
}


/****************************
***** FEATURE SELECTION *****
****************************/

pub struct FeatureSelector {
    selected_features: HashMap<GLCMFeature, String>
}

impl FeatureSelector {
    pub fn features_aliases(&self) -> Vec<(GLCMFeature, String)> {
        let mut f:Vec<_> = self.selected_features.iter().map(|(k,v)| (*k,v.clone())).collect();
        f.sort_by_key(|k|k.1.clone());
        f
    }
}

impl Default for FeatureSelector {
    fn default() -> Self {
        let selected_features =
            GLCMFeature::iter().map(|f|{
                (f,f.to_string().replace("_"," "))
            }).collect();
        FeatureSelector {
            selected_features
        }
    }
}

pub fn update_feature_selector(feature_selector:&mut FeatureSelector, ui:&mut Ui) {

    if ui.button("deselect all").clicked() {
        feature_selector.selected_features.clear();
    }

    if ui.button("select all").clicked() {
        for feature in GLCMFeature::iter() {
            feature_selector.selected_features.insert(feature, feature.to_string().replace("_"," "));
        }
    }

    for feature in GLCMFeature::iter() {
        let mut is_selected = feature_selector.selected_features.contains_key(&feature);
        if ui.checkbox(&mut is_selected, feature.to_string().replace("_"," ")).changed() {
            if is_selected {
                feature_selector.selected_features.insert(feature.clone(),feature.to_string().to_lowercase());
            } else {
                feature_selector.selected_features.remove(&feature);
            }
        }
    }
}


/****************************
******* MAP OPTIONS *******
****************************/

pub struct MapOptSelector {
    kernel_radius: usize,
    num_bins: usize,
    kernel_radius_buf: String,
    num_bins_buf: String,
}

impl Default for MapOptSelector {
    fn default() -> Self {
        MapOptSelector {
            kernel_radius: 1,
            num_bins: 32,
            kernel_radius_buf: String::new(),
            num_bins_buf: String::new()
        }
    }
}

pub fn update_map_options(map_opts:&mut MapOptSelector, ctx: &Context, ui: &mut Ui) {
    ui.horizontal(|ui|{
        ui.label(format!("Kernel Radius: [{}]\t ",map_opts.kernel_radius));
        let te = egui::TextEdit::singleline(&mut map_opts.kernel_radius_buf).desired_width(40.0);
        let h = ui.add(te);
        if h.lost_focus() {
            if let Ok(parsed) = map_opts.kernel_radius_buf.parse::<i32>() {
                if parsed == 0 {
                    map_opts.kernel_radius = 1;
                }else {
                    map_opts.kernel_radius = parsed.abs() as usize;
                }
            }
        }
    });

    ui.horizontal(|ui|{

        ui.label(format!("Number of Bins: [{}]\t ",map_opts.num_bins));
        let te = egui::TextEdit::singleline(&mut map_opts.num_bins_buf).desired_width(40.0);
        let h = ui.add(te);
        if h.lost_focus() {
            if let Ok(parsed) = map_opts.num_bins_buf.parse::<i32>() {
                if parsed < 1 {
                    map_opts.num_bins = 4;
                }else {
                    map_opts.num_bins = parsed.abs() as usize;
                }
            }
        }
    });

}

/****************************
***** OUTPUT SELECTION ******
****************************/
pub struct OutputSelector {
    output_dir_buf: String,
    output_dir: Option<PathBuf>,
    output_dir_dialog: FileDialog,
    handle: Option<JoinHandle<bool>>,
    is_writing_output: bool,
    is_complete: bool,
}

impl Default for OutputSelector {
    fn default() -> Self {
        Self {
            output_dir_buf: String::new(),
            output_dir: None,
            output_dir_dialog: FileDialog::new(),
            handle: None,
            is_writing_output: false,
            is_complete: false,
        }
    }
}

pub fn update_output_selector(output_selector:&mut OutputSelector, input_selector: &InputSelector, features:&FeatureSelector, launcher: &mut GLCMLauncher, ctx: &Context, ui: &mut Ui) {

    ui.horizontal(|ui|{
        ui.label("Output Directory:");
        if output_selector.output_dir.is_some() {
            ui.label(RichText::new("✅").color(Color32::GREEN));
        }else {
            ui.label(RichText::new("x").color(Color32::RED));
        }

        let h = ui.text_edit_singleline(&mut output_selector.output_dir_buf);

        if ui.button("browse").clicked() {
            output_selector.output_dir_dialog.pick_directory()
        }

        if h.lost_focus() {
            output_selector.output_dir = None;
            let p = Path::new(&output_selector.output_dir_buf);
            if p.exists() {
                output_selector.output_dir = Some(p.to_path_buf());
            }
        }

    });

    output_selector.output_dir_dialog.update(ctx);

    if let Some(path) = output_selector.output_dir_dialog.take_picked() {
        output_selector.output_dir_buf = path.display().to_string();
        output_selector.output_dir = Some(path);
    }

    // try to write output volumes
    if let Some(output_dir) = &output_selector.output_dir {

        if let Some(results) = launcher.result.take() {

            output_selector.is_writing_output = false;
            output_selector.is_complete = false;

            let header = launcher.ref_header.take().unwrap();

            let (data,dims) = results;

            let input_path = input_selector.volume_path.as_ref().unwrap().to_path_buf();
            let file_stem = input_path.file_stem().unwrap().to_str().unwrap().to_string();

            let feature_aliases = features.features_aliases();
            let t_output_dir = output_dir.to_path_buf();
            let h = std::thread::spawn(move || {

                let vol_stride:usize = dims.shape_ns()[0..3].iter().product();
                for (f,alias) in feature_aliases {
                    let i = f as usize;
                    let vol = &data[i*vol_stride..(i+1) * vol_stride];
                    let path = t_output_dir.join(format!("{}{}{}",file_stem,"_",alias));
                    let vol_dims = ArrayDim::from_shape(&dims.shape()[0..3]);
                    match &header {
                        Header::Nrrd(nhdr) => write_nrrd(path,vol,vol_dims,Some(nhdr),false, Encoding::raw),
                        Header::Nifti(nii) => write_nifti_with_header(path,vol,vol_dims,nii)
                    };
                }
                true

            });
            output_selector.is_writing_output = true;
            output_selector.handle = Some(h);
        }
    }


    if let Some(h) = output_selector.handle.take() {
        if h.is_finished() {
            output_selector.is_writing_output = false;
            output_selector.is_complete = true;
        }else {
            output_selector.handle = Some(h);
        }
    }

    if output_selector.is_writing_output {
        ui.label("writing output ...");
    }

    if output_selector.is_complete {
        ui.label("writing complete");
    }

}


/***************************
****** DATA SELECTION ******
****************************/
pub struct InputSelector {
    /// buffer to hold the volume path ui
    volume_path_buf: String,
    /// buffer to old the mask path ui
    mask_path_buf: String,
    /// validated volume path
    volume_path: Option<PathBuf>,
    /// validated mask path
    mask_path: Option<PathBuf>,

    /// file dialog box objects
    volume_file_dialog: FileDialog,
    mask_file_dialog: FileDialog,
}

pub fn update_data_loader(data_loader:&mut InputSelector, ctx: &Context, ui: &mut Ui) {

    ui.horizontal(|ui|{
        ui.label("Input Volume:");
        if data_loader.volume_path.is_some() {
            ui.label(RichText::new("✅").color(Color32::GREEN));
        }else {
            ui.label(RichText::new("x").color(Color32::RED));
        }

        let h = ui.text_edit_singleline(&mut data_loader.volume_path_buf);

        if ui.button("browse").clicked() {
            data_loader.volume_file_dialog.pick_file();
        }

        if h.lost_focus() {
            data_loader.volume_path = None;
            let p = Path::new(&data_loader.volume_path_buf);
            if p.exists() {
                data_loader.volume_path = Some(p.to_path_buf());
            }
        }
    });

    data_loader.volume_file_dialog.update(ctx);

    ui.horizontal(|ui|{
        ui.label("Input Mask:");
        if data_loader.mask_path.is_some() {
            ui.label(RichText::new("✅").color(Color32::GREEN));
        }

        let h = ui.text_edit_singleline(&mut data_loader.mask_path_buf);

        if ui.button("browse").clicked() {
            data_loader.mask_file_dialog.pick_file();
        }

        if h.lost_focus() {
            data_loader.mask_path = None;
            let p = Path::new(&data_loader.mask_path_buf);
            if p.exists() {
                data_loader.mask_path = Some(p.to_path_buf());
            }
        }
    });

    data_loader.mask_file_dialog.update(ctx);

    // ui.horizontal(|ui|{
    //     ui.label("Selected Volume:\t");
    //     if let Some(vol_path) = &data_loader.volume_path {
    //         ui.label(vol_path.display().to_string());
    //     }
    // });
    //
    // ui.horizontal(|ui| {
    //     ui.label("Selected Mask:\t");
    //     if let Some(mask_path) = &data_loader.mask_path {
    //         ui.label(mask_path.display().to_string());
    //     }
    // });

    if let Some(path) = data_loader.volume_file_dialog.take_picked() {
        data_loader.volume_path_buf = path.display().to_string();
        data_loader.volume_path = Some(path);
    }

    if let Some(path) = data_loader.mask_file_dialog.take_picked() {
        data_loader.mask_path_buf = path.display().to_string();
        data_loader.mask_path = Some(path);
    }

}

impl Default for InputSelector {
    fn default() -> Self {
        InputSelector {
            volume_path_buf: String::new(),
            mask_path_buf: String::new(),
            volume_path: None,
            mask_path: None,
            volume_file_dialog: FileDialog::new(),
            mask_file_dialog: FileDialog::new(),
        }
    }
}
