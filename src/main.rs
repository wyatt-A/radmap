use std::collections::HashMap;
use strum::IntoEnumIterator;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use eframe::{egui, Frame, NativeOptions};
use eframe;
use eframe::egui::{Context, ProgressBar, Ui};
use egui_file_dialog::FileDialog;
use glcm::run_glcm_map;
use glcm::ui::MapOpts;
use glcm::glcm::GLCMFeature;
use array_lib::{io_nrrd, ArrayDim};

fn main() {

    let mut native_options = NativeOptions::default();
    native_options.vsync = true;

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
            ui.heading("GLCM RadMapper 9000");

            update_data_loader(&mut self.data_loader, ctx, ui);
            update_output_selector(&mut self.output_selector, ctx, ui);
            update_map_options(&mut self.opts_selector, ctx, ui);
            update_feature_selector(&mut self.feature_selector, ui);
            update_progress(&mut self.progress, ui);
            update_glcm_launcher(
                &mut self.map_opts, &mut self.progress, &mut self.glcm_launcher,&self.opts_selector,
                &self.feature_selector, &self.data_loader,&self.output_selector,ui
            );

        });
    }
}

pub fn update_options(opts:&mut MapOpts, map_opts:&MapOptSelector, features:&FeatureSelector) {

    opts.kernel_radius = map_opts.kernel_radius;
    opts.features = features.selected_features.clone();
    opts.n_bins = map_opts.num_bins;
    opts.separator = None;

}



pub struct GLCMLauncher {
    glcm_calc_handle: Option<JoinHandle<(Vec<f64>, ArrayDim)>>
}

impl Default for GLCMLauncher {
    fn default() -> Self {
        GLCMLauncher {
            glcm_calc_handle: None
        }
    }
}

pub fn update_glcm_launcher(map_opts:&mut MapOpts, progress:&mut Progress, launcher: &mut GLCMLauncher, opts_selector:&MapOptSelector, features:&FeatureSelector, data_selector:&InputSelector, output_selector: &OutputSelector, ui:&mut Ui) {

    // check that files have been selected
    if data_selector.volume_path.is_some() && output_selector.output_dir.is_some() {

        update_options(map_opts,opts_selector,features);

        if ui.button("LAUNCH").clicked() {

            let vol_path = data_selector.volume_path.as_ref().unwrap().clone();
            let vol_handle = std::thread::spawn(move||{
                io_nrrd::read_nrrd(vol_path)
            });

            let mask_handle = if let Some(mask_path) =  &data_selector.mask_path {
                let mp = mask_path.clone();
                let h = std::thread::spawn(move||{
                    io_nrrd::read_nrrd::<f64>(mp)
                });
                Some(h)
            }else {
                None
            };

            let (vol, vol_dims, vol_header) = vol_handle.join().expect("failed to retrieve volume from loader thread");
            let mask = mask_handle.map(|h| h.join().expect("failed to retrieve mask from loader thread"));

            progress.total_vox_to_compute = Some(vol_dims.numel());

            let t_map_opts = map_opts.clone();
            let t_progress = progress.progress.clone();
            let glcm_calc_handle = std::thread::spawn(move ||{
                // check that the mask and volume have compatible shapes
                let mask = mask.map(|(mask_data,mask_dims,_)| {
                    assert_eq!(mask_dims.shape_ns(), vol_dims.shape_ns(), "mask and volume have different shapes");
                    mask_data
                });
                run_glcm_map(t_map_opts,vol,mask,vol_dims,t_progress)
            });

            launcher.glcm_calc_handle = Some(glcm_calc_handle);

        }
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
        ui.label("Kernel Radius:\t");
        let h = ui.text_edit_singleline(&mut map_opts.kernel_radius_buf);
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
        ui.label("current value:\t");
        ui.label(map_opts.kernel_radius.to_string());
    });

    ui.horizontal(|ui|{
        ui.label("Number of Bins:\t");
        let h = ui.text_edit_singleline(&mut map_opts.num_bins_buf);
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

    ui.horizontal(|ui|{
        ui.label("current value:\t");
        ui.label(map_opts.num_bins.to_string());
    });

}

/****************************
***** OUTPUT SELECTION ******
****************************/
pub struct OutputSelector {
    output_dir_buf: String,
    output_dir: Option<PathBuf>,
    output_dir_dialog: FileDialog,
}

impl Default for OutputSelector {
    fn default() -> Self {
        Self {
            output_dir_buf: String::new(),
            output_dir: None,
            output_dir_dialog: FileDialog::new(),
        }
    }
}

pub fn update_output_selector(output_selector:&mut OutputSelector, ctx: &Context, ui: &mut Ui) {

    ui.horizontal(|ui|{
        ui.label("Output Directory:\t");

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

    ui.horizontal(|ui|{
        ui.label("Selected Directory:\t");
        if let Some(vol_path) = &output_selector.output_dir {
            ui.label(vol_path.display().to_string());
        }
    });

    if let Some(path) = output_selector.output_dir_dialog.take_picked() {
        output_selector.output_dir_buf = path.display().to_string();
        output_selector.output_dir = Some(path);
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
        ui.label("Input Volume:\t");

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
        ui.label("Input Mask:\t");

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

    ui.horizontal(|ui|{
        ui.label("Selected Volume:\t");
        if let Some(vol_path) = &data_loader.volume_path {
            ui.label(vol_path.display().to_string());
        }
    });

    ui.horizontal(|ui| {
        ui.label("Selected Mask:\t");
        if let Some(mask_path) = &data_loader.mask_path {
            ui.label(mask_path.display().to_string());
        }
    });

    if let Some(path) = data_loader.volume_file_dialog.take_picked() {
        data_loader.volume_path_buf = path.display().to_string();
        data_loader.volume_path = Some(path);
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
