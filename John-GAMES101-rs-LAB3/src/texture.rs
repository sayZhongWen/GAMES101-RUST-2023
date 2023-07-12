use nalgebra::Vector3;
use opencv::core::{MatTraitConst, VecN};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub struct Texture {
    pub img_data: opencv::core::Mat,
    pub width: usize,
    pub height: usize,
}

impl Texture {
    pub fn new(name: &str) -> Self {
        let img_data = imread(name, IMREAD_COLOR).expect("Image reading error!");
        let width = img_data.cols() as usize;
        let height = img_data.rows() as usize;
        Texture {
            img_data,
            width,
            height,
        }
    }

    pub fn get_color(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        if u < 0.0 {
            u = 0.0;
        }
        if u > 1.0 {
            u = 1.0;
        }
        if v < 0.0 {
            v = 0.0;
        }
        if v > 1.0 {
            v = 1.0;
        }

        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let color: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32).unwrap();

        Vector3::new(color[2] as f64, color[1] as f64, color[0] as f64)
    }

    pub fn getColorBilinear(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        if u < 0.0 {
            u = 0.0;
        }
        if u > 1.0 {
            u = 1.0;
        }
        if v < 0.0 {
            v = 0.0;
        }
        if v > 1.0 {
            v = 1.0;
        }
        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let color00: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32).unwrap();
        let color01: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32+1).unwrap();
        let color10: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32+1, u_img as i32).unwrap();
        let color11: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32+1, u_img as i32+1).unwrap();
        let u00=Vector3::new(color00[0] as f64,color00[1] as f64,color00[2] as f64);
        let u01=Vector3::new(color01[0] as f64,color01[1] as f64,color01[2] as f64);
        let u10=Vector3::new(color10[0] as f64,color10[1] as f64,color10[2] as f64);
        let u11=Vector3::new(color11[0] as f64,color11[1] as f64,color11[2] as f64);
        let s=v_img-v_img as i32 as f64;
        let t=u_img-u_img as i32 as f64;
        let u0=lerp(s,&u00,&u10);
        let u1=lerp(s,&u01,&u11);
        let color=lerp(t,&u0,&u1);
        Vector3::new(color[2] as f64, color[1] as f64, color[0] as f64)
    }
}
fn lerp<'a>(x:f64,u0:&'a Vector3<f64>,u1:&'a Vector3<f64>)-> Vector3<f64>{
    u0+x*(u1-u0)
}