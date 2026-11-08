//! Camera motion compensation (CMC) shared across trackers.
//!
//! A [`CameraMotion`] is a 2x3 affine transform that maps coordinates from the
//! previous frame into the current frame. Callers estimate it however they like
//! (for example image registration with OpenCV in a demo) and pass it to a
//! tracker's `update`; the tracker warps its predicted track states so motion
//! association stays valid under a moving camera. Trackers that consume it (Deep
//! OC-SORT today, BoT-SORT and StrongSORT++ in the future) all apply it the same
//! way through [`super::KalmanTrack::apply_camera_motion`].

use crate::utils::kalman::{BoxState, CovarianceMatrix, MeasurementVector, StateVector};

/// A 2x3 affine camera-motion transform `[[a, b, tx], [c, d, ty]]`.
///
/// Maps a previous-frame point `(x, y)` to `(a*x + b*y + tx, c*x + d*y + ty)`.
/// The default is the identity (no camera motion).
#[derive(Debug, Clone, Copy)]
pub struct CameraMotion {
    /// Row 0 of the linear part.
    pub a: f32,
    pub b: f32,
    /// Horizontal translation.
    pub tx: f32,
    /// Row 1 of the linear part.
    pub c: f32,
    pub d: f32,
    /// Vertical translation.
    pub ty: f32,
}

impl Default for CameraMotion {
    fn default() -> Self {
        Self::identity()
    }
}

impl CameraMotion {
    /// Build a transform from the six affine coefficients.
    pub fn new(a: f32, b: f32, tx: f32, c: f32, d: f32, ty: f32) -> Self {
        Self { a, b, tx, c, d, ty }
    }

    /// The identity transform (no camera motion).
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Whether this transform is the identity, so application can be skipped.
    pub fn is_identity(&self) -> bool {
        self.a == 1.0
            && self.b == 0.0
            && self.tx == 0.0
            && self.c == 0.0
            && self.d == 1.0
            && self.ty == 0.0
    }

    /// Uniform scale factor implied by the linear part (`sqrt(|det|)`).
    fn scale(&self) -> f32 {
        (self.a * self.d - self.b * self.c).abs().sqrt()
    }

    /// 8x8 linear warp matrix for a Kalman state laid out as `box_state`.
    ///
    /// The 2x2 linear part applies to the position `(x, y)` and velocity
    /// `(vx, vy)` blocks. In [`BoxState::AspectHeight`] the uniform scale applies
    /// to the height `h` and its velocity, and the aspect ratio stays unchanged;
    /// in [`BoxState::WidthHeight`] the linear part also applies to `(w, h)` and
    /// its velocity, as the reference BoT-SORT warps its state.
    fn linear_state_matrix(&self, box_state: BoxState) -> CovarianceMatrix {
        let mut r = CovarianceMatrix::identity();
        let linear_blocks: &[usize] = match box_state {
            BoxState::AspectHeight => {
                let scale = self.scale();
                r[(3, 3)] = scale;
                r[(7, 7)] = scale;
                &[0, 4]
            }
            BoxState::WidthHeight => &[0, 2, 4, 6],
        };
        for &i in linear_blocks {
            r[(i, i)] = self.a;
            r[(i, i + 1)] = self.b;
            r[(i + 1, i)] = self.c;
            r[(i + 1, i + 1)] = self.d;
        }
        r
    }

    /// Warp a Kalman mean and covariance, laid out as `box_state`, in place.
    pub fn apply_state(
        &self,
        mean: &mut StateVector,
        covariance: &mut CovarianceMatrix,
        box_state: BoxState,
    ) {
        let r = self.linear_state_matrix(box_state);
        let mut warped = r * *mean;
        warped[0] += self.tx;
        warped[1] += self.ty;
        *mean = warped;
        *covariance = r * *covariance * r.transpose();
    }

    /// Warp an XYAH observation `[cx, cy, aspect, height]` in place.
    pub fn apply_observation(&self, obs: &mut MeasurementVector) {
        let (cx, cy) = (obs[0], obs[1]);
        obs[0] = self.a * cx + self.b * cy + self.tx;
        obs[1] = self.c * cx + self.d * cy + self.ty;
        obs[3] *= self.scale();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        assert!(CameraMotion::default().is_identity());
    }

    #[test]
    fn identity_is_a_no_op() {
        let cmc = CameraMotion::identity();
        assert!(cmc.is_identity());
        let mut mean = StateVector::from_vec(vec![10.0, 20.0, 0.5, 40.0, 1.0, 2.0, 0.0, 0.0]);
        let original = mean;
        let mut cov = CovarianceMatrix::identity();
        cmc.apply_state(&mut mean, &mut cov, BoxState::AspectHeight);
        assert!((mean - original).norm() < 1e-5);
    }

    #[test]
    fn translation_shifts_position_not_velocity() {
        let cmc = CameraMotion::new(1.0, 0.0, 5.0, 0.0, 1.0, -3.0);
        assert!(!cmc.is_identity());
        let mut mean = StateVector::from_vec(vec![10.0, 20.0, 0.5, 40.0, 1.0, 2.0, 0.0, 0.0]);
        let mut cov = CovarianceMatrix::identity();
        cmc.apply_state(&mut mean, &mut cov, BoxState::AspectHeight);
        assert!((mean[0] - 15.0).abs() < 1e-5);
        assert!((mean[1] - 17.0).abs() < 1e-5);
        // Pure translation leaves velocity untouched.
        assert!((mean[4] - 1.0).abs() < 1e-5);
        assert!((mean[5] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn scale_grows_height_and_velocity() {
        // 2x uniform scale: positions and height double, velocity scales linearly.
        let cmc = CameraMotion::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        let mut mean = StateVector::from_vec(vec![10.0, 20.0, 0.5, 40.0, 1.0, 2.0, 0.0, 3.0]);
        let mut cov = CovarianceMatrix::identity();
        cmc.apply_state(&mut mean, &mut cov, BoxState::AspectHeight);
        assert!((mean[0] - 20.0).abs() < 1e-5);
        assert!((mean[3] - 80.0).abs() < 1e-5); // height doubled
        assert!((mean[4] - 2.0).abs() < 1e-5); // vx doubled
        assert!((mean[7] - 6.0).abs() < 1e-5); // vh doubled
    }

    #[test]
    fn scale_grows_width_and_height_in_a_width_height_state() {
        let cmc = CameraMotion::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        let mut mean = StateVector::from_vec(vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 0.5, 3.0]);
        let mut cov = CovarianceMatrix::identity();
        cmc.apply_state(&mut mean, &mut cov, BoxState::WidthHeight);
        let doubled = StateVector::from_vec(vec![20.0, 40.0, 60.0, 80.0, 2.0, 4.0, 1.0, 6.0]);
        assert!((mean - doubled).norm() < 1e-5);
    }

    #[test]
    fn observation_is_warped() {
        let cmc = CameraMotion::new(1.0, 0.0, 4.0, 0.0, 1.0, 2.0);
        let mut obs = MeasurementVector::from_vec(vec![10.0, 20.0, 0.5, 40.0]);
        cmc.apply_observation(&mut obs);
        assert!((obs[0] - 14.0).abs() < 1e-5);
        assert!((obs[1] - 22.0).abs() < 1e-5);
        assert!((obs[3] - 40.0).abs() < 1e-5);
    }
}
