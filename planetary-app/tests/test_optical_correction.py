"""Correct Optical Aberrations: a moon as the PSF, auto selection and the panel."""
import os
import re
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication
from scipy import ndimage

from planetary_tools.core.document import ImageDocument
from planetary_tools.filters import optical_correction as oc
from planetary_tools.filters.registry import FILTERS, apply_filter
from planetary_tools.ui.main_window import MainWindow
from planetary_tools.ui.optical_correction_dialog import OpticalCorrectionDialog

# Astigmatism-like blur, wider horizontally.
SIGMA_Y, SIGMA_X = 1.8, 3.0
MOON = (250, 60)
FAINT_MOON = (285, 150)
TITAN = (215, 165)


def scene(sigma=(SIGMA_Y, SIGMA_X), seed=3, moons=True, shift=(0, 0)):
    h, w = 200, 320
    sharp = np.zeros((h, w))
    yy, xx = np.mgrid[:h, :w]
    sharp[np.hypot(yy - 100, xx - 90) < 40] = .5
    sharp[(np.abs(yy - 100) < 3) & (np.abs(xx - 90) < 70)] = .35  # ring
    if moons:
        sharp[MOON[1] + shift[1], MOON[0] + shift[0]] = 2.0
        sharp[FAINT_MOON[1] + shift[1], FAINT_MOON[0] + shift[0]] = .6
        # A resolved disc, brighter than any moon: not a point source.
        disc = np.hypot(yy - TITAN[1], xx - TITAN[0]) < 3.5
        sharp[disc] = 8.0/disc.sum()
    blurred = ndimage.gaussian_filter(sharp, sigma)
    rng = np.random.default_rng(seed)
    return (blurred + 1e-4 + rng.normal(0, 1e-5, blurred.shape)).astype(np.float32)


def measured(plane, x, y):
    m = oc.measure_psf(plane, x, y)
    return m.fwhm_major, m.fwhm_minor


class FilterTests(unittest.TestCase):
    def test_auto_selects_bright_compact_moon_not_resolved_disc(self):
        image = scene()
        choice = oc.auto_select_moon(image, True)
        self.assertLess(np.hypot(choice.x - MOON[0], choice.y - MOON[1]), 1.0)
        m = oc.measure_choice(image, True, choice)
        # Measured shape: long axis horizontal, matching the blur.
        self.assertAlmostEqual(m.fwhm_major, 2.355*SIGMA_X, delta=.6)
        self.assertAlmostEqual(m.fwhm_minor, 2.355*SIGMA_Y, delta=.5)
        self.assertLess(abs(m.angle_deg), 10)

    def test_auto_ignores_a_faint_quantisation_blip(self):
        # 16-bit stacks have flat quantised sky; a 2-step blip measured
        # deceptively compact next to wide raw-stack moons (2026-11-21 1055_7).
        h, w = 200, 320
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 100, xx - 90) < 40] = .5
        sharp[MOON[1], MOON[0]] = .4
        sharp[FAINT_MOON[1], FAINT_MOON[0]] = .15
        image = ndimage.gaussian_filter(sharp, (4.0, 5.5)) + 1e-4
        image[np.hypot(yy - 25, xx - 40) < 5] += 1/65535
        image[25, 40] += 2/65535
        image = (np.round(image*65535)/65535).astype(np.float32)
        choice = oc.auto_select_moon(image, True)
        self.assertLess(min(np.hypot(choice.x - x, choice.y - y) for x, y in (MOON, FAINT_MOON)), 1.5)

    def test_correction_makes_every_moon_round_at_the_minor_width(self):
        image = scene()
        # Max boost limits how far a strong (1.67:1) elongation is undone.
        # Unprotected: the full correction everywhere, so the strength shows.
        for boost, ratio in ((4.0, 1.4), (12.0, 1.25)):
            out = oc.correct_optical_aberrations(image, True, max_boost=boost,
                                                 protect_background=False)
            for x, y in (MOON, FAINT_MOON):
                before = measured(image, x, y)
                major, minor = measured(out, x, y)
                self.assertGreater(before[0]/before[1], 1.5)
                self.assertLess(major/minor, ratio, (boost, x, y))
                # Round at the narrow width, not blurred to the wide one.
                self.assertLess(major, before[0]*.85)
                self.assertLess(minor, before[1]*1.08)
        # Flux and sky level are preserved.
        self.assertAlmostEqual(float(out.sum())/float(image.sum()), 1.0, places=3)
        self.assertAlmostEqual(float(np.median(out[:20, 200:])),
                               float(np.median(image[:20, 200:])), delta=2e-5)

    def test_no_rings_around_a_planet_far_brighter_than_the_moon(self):
        # The moon's faint halo is measured far less accurately than the
        # planet's glow is bright. Reshaping it left dark rings around the
        # planet (2026-11-21 L3 stacks: 0.69% of the peak at 1x boost).
        h, w = 260, 360
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 130, xx - 120) < 50] = 1.0
        sharp[60, 290] = .03  # moon peak ~0.1% of the planet's
        image = (.75*ndimage.gaussian_filter(sharp, (1.8, 3.0))
                 + .25*ndimage.gaussian_filter(sharp, 9.0) + 2e-4
                 + np.random.default_rng(7).normal(0, 1e-5, (h, w))).astype(np.float32)
        r = np.hypot(yy - 130, xx - 120)
        angle = np.arctan2(yy - 130, xx - 120)

        def deepest_dip(img):
            """Worst dip below the brightness further out, % of the peak."""
            worst = 0.0
            for a0 in np.linspace(-np.pi, np.pi, 72, endpoint=False):
                sel = (np.abs(np.angle(np.exp(1j*(angle - a0)))) < np.pi/72) & (r > 50) & (r < 110)
                idx = r[sel].astype(int) - 50
                prof = np.bincount(idx, img[sel], 60)/np.maximum(np.bincount(idx, minlength=60), 1)
                worst = max(worst, float((np.maximum.accumulate(prof[::-1])[::-1] - prof).max()))
            return worst/float(img.max())*100

        before = deepest_dip(image)
        for boost in (1.0, 4.0):
            out = oc.correct_optical_aberrations(image, True, max_boost=boost)
            # Edge overshoot from sharpening the long axis stays tiny; the
            # halo-reshaping version reached 0.149 here.
            self.assertLess(deepest_dip(out), max(3*before, .01), boost)
        major, minor = measured(out, 290, 60)
        start = measured(image, 290, 60)
        self.assertLess(major/minor, start[0]/start[1] - .15)

    def test_never_smooths_the_noise_and_one_times_is_no_change(self):
        # Dividing by the noisy measured spectrum fell towards zero at high
        # frequencies: speckle around the moon was blurred, the moon left
        # elongated.
        image = scene()
        same = oc.correct_optical_aberrations(image, True, max_boost=1.0,
                                              coma=False, trefoil=False)
        np.testing.assert_allclose(same, image, atol=1e-6)
        sky = (slice(0, 25), slice(180, 320))
        for boost in (2.0, 4.0):
            kept = oc.correct_optical_aberrations(image, True, max_boost=boost)
            self.assertGreaterEqual(float(kept[sky].std()), .99*float(image[sky].std()))
            out = oc.correct_optical_aberrations(image, True, max_boost=boost,
                                                 protect_background=False)
            self.assertGreater(float(out[sky].std()), float(image[sky].std()))
            major, minor = measured(out, *MOON)
            self.assertLess(major/minor, measured(image, *MOON)[0]/measured(image, *MOON)[1] - .15)

    def test_protection_keeps_a_faint_moons_glow_free_of_speckle(self):
        # One boost everywhere amplified the noise in a moon's faint outer
        # glow, so the corrected moon looked speckled (2026-11-21 IR642).
        image = scene()
        yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
        r = np.hypot(xx - FAINT_MOON[0], yy - FAINT_MOON[1])
        glow = (r > 10) & (r < 20)

        def fine(z):
            return z - ndimage.gaussian_filter(z, 1.0)
        for boost in (4.0, 12.0):
            plain = oc.correct_optical_aberrations(image, True, max_boost=boost,
                                                   protect_background=False)
            kept = oc.correct_optical_aberrations(image, True, max_boost=boost)
            base = float(fine(image)[glow].std())
            self.assertGreater(float(fine(plain)[glow].std()), 3*base)
            self.assertLess(float(fine(kept)[glow].std()), 1.1*base)
            # The bright moon is still corrected in full.
            np.testing.assert_allclose(measured(kept, *MOON), measured(plain, *MOON), atol=.1)

    def test_dark_gap_beside_the_planet_gets_the_planets_boost(self):
        # Measured against a local median raised by the planet's glow, the
        # gaps between rings and globe read as empty and got no boost beside
        # a fully boosted limb; sharpening turned the seams into dark hooks
        # (2026-11-21 IR642). The gap must match the unprotected correction.
        h, w = 220, 360
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 110, xx - 110) < 45] = .5
        sharp[(np.abs(yy - 110) < 6) & (xx > 160) & (xx < 215)] = .35
        sharp[60, 300] = 2.0
        sharp[170, 320] = 1.0
        image = (ndimage.gaussian_filter(sharp, (SIGMA_Y, SIGMA_X)) + 1e-4
                 + np.random.default_rng(5).normal(0, 1e-5, (h, w))).astype(np.float32)
        kept = oc.correct_optical_aberrations(image, True, coma=False, trefoil=False)
        plain = oc.correct_optical_aberrations(image, True, coma=False, trefoil=False,
                                               protect_background=False)
        gap = (slice(100, 121), slice(154, 162))
        # The local-median version differed by 7.5% of the peak here.
        self.assertLess(float(np.abs(kept[gap] - plain[gap]).max()), .002*float(image.max()))

    def test_protect_background_leaves_sky_noise_alone(self):
        image = scene()
        sky = (slice(0, 25), slice(180, 320))
        plain = oc.correct_optical_aberrations(image, True, protect_background=False)
        kept = oc.correct_optical_aberrations(image, True, protect_background=True)
        self.assertGreater(float(plain[sky].std()), 1.2*float(image[sky].std()))
        self.assertLess(float(kept[sky].std()), 1.05*float(image[sky].std()))
        for x, y in (MOON, FAINT_MOON):
            np.testing.assert_allclose(measured(kept, x, y), measured(plain, x, y), atol=.1)

    def lobed_scene(self, lobes, seed=11):
        """Moons blurred by a round core plus off-centre lobes (coma: one,
        trefoil: three), as in the 2026-11-01 and 2026-11-03 stacks."""
        h, w = 200, 320
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 100, xx - 90) < 40] = .5
        for x, y, f in ((250, 60, 2.0), (285, 150, 1.2), (215, 170, .9)):
            sharp[y, x] = f
        psf = np.zeros((41, 41))
        py, px = np.mgrid[-20:21, -20:21]
        psf += np.exp(-(px**2 + py**2)/(2*2.2**2))
        for k in range(lobes):
            a = np.radians(30 + 360*k/lobes)
            psf += .45*np.exp(-((px - 4*np.cos(a))**2 + (py - 4*np.sin(a))**2)/(2*1.6**2))
        image = ndimage.convolve(sharp, psf/psf.sum(), mode='reflect') + 1e-4
        image += np.random.default_rng(seed).normal(0, 5e-6, (h, w))
        return image.astype(np.float32)

    def test_coma_and_trefoil_are_measured_and_corrected_separately(self):
        from planetary_tools.filters.optical_correction import _harmonic_vectors
        for lobes, order, key in ((1, 1, 'coma'), (3, 3, 'trefoil')):
            image = self.lobed_scene(lobes)
            moons = oc.auto_select_moons(image, True)
            self.assertGreater(len(moons), 1)
            m = oc.measure_moons(image, True, moons)
            amount = {1: m.coma_amount, 3: m.trefoil_amount}[order]
            other = {1: m.trefoil_amount, 3: m.coma_amount}[order]
            self.assertGreater(amount, .08, key)
            if key == 'trefoil':
                # Three equal lobes have no one-sided part; a single tail
                # (like real coma) also has a three-fold part.
                self.assertLess(other, amount/3, key)
            self.assertGreater(m.agreement[order], .9)
            self.assertEqual(oc.disagreements(m), [])
            off = oc.correct_optical_aberrations(image, True, astigmatism=False, coma=False,
                                                 trefoil=False)
            np.testing.assert_allclose(off, image, atol=1e-6)
            only = oc.correct_optical_aberrations(
                image, True, astigmatism=False, coma=key == 'coma', trefoil=key == 'trefoil')
            for x, y in ((250, 60), (285, 150)):
                before = abs(_harmonic_vectors(oc.measure_psf(image, x, y).psf, m.fwhm_minor)[order])
                after = abs(_harmonic_vectors(oc.measure_psf(only, x, y).psf, m.fwhm_minor)[order])
                self.assertLess(after, before/2.5, (key, x, y))

    def test_correction_follows_a_long_collimation_tail(self):
        # A narrow tail reaching ~3 FWHM, as on an SCT out of collimation
        # (2026-11-01 1205_8): the phase is corrected out to its tip.
        h, w = 200, 320
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 100, xx - 90) < 40] = .5
        for x, y, f in ((250, 60, 2.0), (285, 150, 1.2), (215, 170, .9)):
            sharp[y, x] = f
        py, px = np.mgrid[-20:21, -20:21]
        psf = np.exp(-(px**2 + py**2)/(2*1.8**2))
        for t in np.arange(2, 15, 0.5):
            psf += .06*np.exp(-t/7)*np.exp(-((px + t*.707)**2 + (py - t*.707)**2)/(2*1.2**2))
        image = ndimage.convolve(sharp, psf/psf.sum(), mode='reflect') + 1e-4
        image = (image + np.random.default_rng(7).normal(0, 5e-6, (h, w))).astype(np.float32)
        m = oc.measure_moons(image, True, oc.auto_select_moons(image, True))
        self.assertGreater(m.flare_radius, 2.5*m.fwhm_major)

        def one_sided(img, x=250, y=60, r=20):
            c = img[y-r:y+r+1, x-r:x+r+1].astype(np.float64) - 1e-4
            d = np.hypot(*np.mgrid[-r:r+1, -r:r+1])
            return np.abs(0.5*(c - c[::-1, ::-1]))[(d >= 4) & (d < 12)].sum()/c.clip(0).sum()

        def corrected(mm):
            return oc.correct_plane(image, mm, *oc.signal_maps(image, mm), target_size=100,
                                    max_boost=3, astigmatism=False, coma=True, trefoil=True)
        core_only = one_sided(corrected(replace(m, flare_radius=0.0)))
        self.assertLess(one_sided(corrected(m)), .88*core_only)
        # No coma or trefoil: no flare, the cutoff stays at the core.
        self.assertEqual(oc.measure_moons(scene(), True, oc.auto_select_moons(scene(), True))
                         .flare_radius, 0.0)

    def test_strength_scales_each_correction(self):
        from planetary_tools.filters.optical_correction import _harmonic_vectors
        # Coma: 0x is no correction, 1x the default, 2x overshoots to a tail
        # the other way.
        image = self.lobed_scene(1)
        fw = oc.measure_moons(image, True, oc.auto_select_moons(image, True)).fwhm_minor

        def coma(img):
            return _harmonic_vectors(oc.measure_psf(img, 250, 60).psf, fw)[1]

        settings = dict(astigmatism=False, trefoil=False)
        zero = oc.correct_optical_aberrations(image, True, coma_strength=0.0, **settings)
        np.testing.assert_allclose(zero, image, atol=1e-6)
        np.testing.assert_array_equal(
            oc.correct_optical_aberrations(image, True, coma_strength=1.0, **settings),
            oc.correct_optical_aberrations(image, True, **settings))
        half = oc.correct_optical_aberrations(image, True, coma_strength=0.5, **settings)
        double = oc.correct_optical_aberrations(image, True, coma_strength=2.0, **settings)
        before = coma(image)
        self.assertLess(abs(coma(half)), .8*abs(before))
        self.assertGreater(abs(coma(half)), .2*abs(before))
        self.assertLess(np.real(coma(double)*np.conj(before)), 0)  # flipped
        # Coma and trefoil together: each strength still applies on its own.
        both = dict(astigmatism=False)
        np.testing.assert_allclose(
            oc.correct_optical_aberrations(image, True, trefoil_strength=0.0, **both),
            oc.correct_optical_aberrations(image, True, trefoil=False, **both), atol=5e-4)
        # Astigmatism: elongation left falls as the strength rises.
        image = scene()
        ratios = []
        for strength in (0.0, 0.5, 1.0, 2.0):
            out = oc.correct_optical_aberrations(
                image, True, coma=False, trefoil=False, protect_background=False,
                max_boost=20, astigmatism_strength=strength)
            major, minor = measured(out, *MOON)
            ratios.append(major/minor)
        self.assertAlmostEqual(ratios[0], measured(image, *MOON)[0]/measured(image, *MOON)[1],
                               delta=.01)
        self.assertTrue(all(a > b for a, b in zip(ratios, ratios[1:])), ratios)

    def test_moons_with_opposite_coma_are_reported(self):
        a = oc.measure_psf(self.lobed_scene(1), 250, 60)
        flipped = replace(a, psf=a.psf[::-1, ::-1].copy(),
                          harmonics={k: (-v if k % 2 else v) for k, v in a.harmonics.items()})
        both = oc.combine_measurements([a, flipped], [1.0, 1.0])
        self.assertLess(both.agreement[1], .2)
        self.assertIn('Coma/collimation', oc.disagreements(both))
        self.assertLess(both.coma_amount, a.coma_amount/3)

    def drift_scene(self, smears, seed=5):
        """Planet and moons sharing an elongated blur, each moon also smeared
        along its own line (length px, angle deg) as drift in a long stack."""
        h, w = 200, 320
        yy, xx = np.mgrid[:h, :w]
        sharp = np.zeros((h, w))
        sharp[np.hypot(yy - 100, xx - 90) < 40] = .5
        for (x, y), (length, angle) in zip(((250, 50), (280, 150), (200, 175)), smears):
            t = np.linspace(-.5, .5, 64)*length
            px = np.round(x + t*np.cos(np.radians(angle))).astype(int)
            py = np.round(y - t*np.sin(np.radians(angle))).astype(int)
            np.add.at(sharp, (py, px), 2.0/t.size)
        blurred = ndimage.gaussian_filter(sharp, (1.8, 2.6))
        rng = np.random.default_rng(seed)
        return (blurred + 1e-4 + rng.normal(0, 1e-5, blurred.shape)).astype(np.float32)

    def test_only_the_elongation_the_moons_share_is_corrected(self):
        # Moons sharing the blur: corrected as measured.
        same = self.drift_scene([(0, 0)]*3)
        m = oc.measure_moons(same, True, oc.auto_select_moons(same, True))
        self.assertEqual(len(m.moons), 3)
        self.assertIsNone(m.boost_ellipse)
        self.assertFalse(oc.elongation_differs(m))
        # Two moons drifted by different amounts: the shared core is the unsmeared
        # moon's, so the boost ignores the drift.
        image = self.drift_scene([(0, 0), (10, 20), (8, 40)])
        m = oc.measure_moons(image, True, oc.auto_select_moons(image, True))
        self.assertEqual(len(m.moons), 3)
        self.assertTrue(oc.elongation_differs(m))
        clean = oc.measure_psf(same, 250, 50)
        sa, sb, angle = m.boost_ellipse
        self.assertAlmostEqual(sa*sa - sb*sb, clean.sigma_major**2 - clean.sigma_minor**2,
                               delta=.25*(clean.sigma_major**2 - clean.sigma_minor**2))
        self.assertLess(abs(angle - clean.angle_deg), 8)
        self.assertLess(m.shared_elongation, .8)
        # The planet gets only the shared correction.
        out = oc.correct_optical_aberrations(image, True, coma=False, trefoil=False)
        full = oc.correct_optical_aberrations(
            image, True, coma=False, trefoil=False, auto=False, moon_x=250, moon_y=50)
        np.testing.assert_allclose(out[60:140, 50:130], full[60:140, 50:130], atol=2e-3)

    def test_smaller_target_also_sharpens(self):
        image = scene()
        full = oc.correct_optical_aberrations(image, True, target_size=100)
        small = oc.correct_optical_aberrations(image, True, target_size=70, max_boost=8)
        self.assertLess(measured(small, *MOON)[1], measured(full, *MOON)[1]*.9)

    def test_manual_choice_follows_a_drifted_moon(self):
        params = {'auto': False, 'moon_x': FAINT_MOON[0] + 2.0, 'moon_y': FAINT_MOON[1] - 1.0}
        drifted = scene(shift=(12, -9))
        choice = oc.choose_moon(drifted, True, params)
        self.assertLess(np.hypot(choice.x - FAINT_MOON[0] - 12, choice.y - FAINT_MOON[1] + 9), 1.0)
        out = apply_filter('optical_correction', drifted, True,
                           {**params, 'max_boost': 12.0, 'protect_background': False})
        major, minor = measured(out, FAINT_MOON[0] + 12, FAINT_MOON[1] - 9)
        self.assertLess(major/minor, 1.25)

    def test_each_colour_channel_uses_its_own_psf(self):
        red = scene(sigma=(1.8, 3.0))
        blue = scene(sigma=(3.0, 1.8), seed=4)
        rgb = np.stack([red, (red + blue)/2, blue], axis=2)
        out = apply_filter('optical_correction', rgb, False,
                           {'per_channel': True, 'clamp': False, 'max_boost': 12.0})
        for c in (0, 2):
            major, minor = measured(out[..., c], *MOON)
            self.assertLess(major/minor, 1.25, c)

    def test_no_moon_raises(self):
        with self.assertRaises(oc.NoMoonError):
            apply_filter('optical_correction', scene(moons=False), True, {})
        with self.assertRaises(oc.NoMoonError):
            apply_filter('optical_correction', scene(), True, {'auto': False})

    def test_registered_for_batch_with_auto_label(self):
        from planetary_tools.batch.pipeline import PipelineStep
        fdef = FILTERS['optical_correction']
        self.assertTrue(fdef.batch_enabled)
        self.assertEqual(fdef.label, 'Correct Optical Aberrations')
        step = PipelineStep('optical_correction', dict(fdef.default_params))
        self.assertTrue(step.label.endswith('Auto'))


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)

        def settings():
            return QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat)
        for target, value in (('planetary_tools.core.presets.PRESET_DIR', root/'presets'),
                              ('planetary_tools.ui.batch_dialog.QSettings', settings),
                              ('planetary_tools.ui.recent_files._settings', settings)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def panel(self):
        panel = OpticalCorrectionDialog(True)
        self.addCleanup(panel.close)
        markers = []
        panel.markers_changed.connect(markers.append)
        panel.set_input_brightness(np.repeat(scene()[..., None], 3, axis=2), True)
        return panel, markers

    def test_auto_shows_the_chosen_moons_and_aberrations(self):
        panel, markers = self.panel()
        self.assertTrue(panel.moon_info.text().startswith('Auto:'))
        self.assertIsNotNone(panel.measured_view.pixmap())
        self.assertIn(MOON, [(round(x), round(y)) for x, y, _ in markers[-1]])
        self.assertRegex(panel._aberration_boxes['astigmatism'].text(), r'Astigmatism \(\d+%\)')
        self.assertTrue(panel.agreement_warning.isHidden())

    def test_drifted_moons_show_the_astigmatism_corrected(self):
        panel = OpticalCorrectionDialog(True)
        self.addCleanup(panel.close)
        image = FilterTests.drift_scene(self, [(0, 0), (10, 20), (8, 40)])
        panel.set_input_brightness(np.repeat(image[..., None], 3, axis=2), True)
        measured, corrected = map(int, re.fullmatch(
            r'Astigmatism \((\d+)%, (\d+)% corrected\)',
            panel._aberration_boxes['astigmatism'].text()).groups())
        self.assertLess(corrected, measured)
        self.assertIn('drift', panel.agreement_warning.text())
        self.assertFalse(panel.agreement_warning.isHidden())

    def test_strength_boxes_scale_the_corrected_figure_and_are_saved(self):
        panel, _ = self.panel()
        box, spin = panel._aberration_boxes['astigmatism'], panel._strength_spins['astigmatism']
        self.assertRegex(box.text(), r'^Astigmatism \(\d+%\)$')
        spin.setValue(2.0)
        measured, corrected = map(int, re.fullmatch(
            r'Astigmatism \((\d+)%, (\d+)% corrected\)', box.text()).groups())
        self.assertAlmostEqual(corrected, 2*measured, delta=1)
        self.assertEqual(panel.get_params()['astigmatism_strength'], 2.0)
        box.setChecked(False)
        self.assertFalse(spin.isEnabled())
        params = panel.get_params()
        other, _ = self.panel()
        other.set_params({**params, 'coma_strength': 0.3})
        self.assertEqual(other._strength_spins['astigmatism'].value(), 2.0)
        self.assertEqual(other._strength_spins['coma'].value(), 0.3)
        self.assertFalse(other._strength_spins['astigmatism'].isEnabled())
        # The filter takes them from saved parameters.
        image = scene()
        np.testing.assert_array_equal(
            apply_filter('optical_correction', image, True,
                         {**params, 'astigmatism': True, 'astigmatism_strength': 0.0}),
            apply_filter('optical_correction', image, True, {**params, 'astigmatism': False}))

    def test_click_or_box_picks_a_moon_and_turns_off_auto(self):
        panel, markers = self.panel()
        changed = []
        panel.params_changed.connect(lambda: changed.append(1))
        panel.set_picked_region(FAINT_MOON[0] + 4, FAINT_MOON[1] + 3, 0, 0)
        params = panel.get_params()
        self.assertFalse(params['auto'])
        self.assertEqual((round(params['moon_x']), round(params['moon_y'])), FAINT_MOON)
        self.assertTrue(changed)
        self.assertEqual([(round(x), round(y)) for x, y, _ in markers[-1]], [FAINT_MOON])
        panel.set_picked_region(MOON[0] - 10, MOON[1] - 10, 20, 20)
        self.assertEqual((round(panel.get_params()['moon_x']),
                          round(panel.get_params()['moon_y'])), MOON)
        # Turning Auto back on chooses again.
        panel.auto.setChecked(True)
        self.assertTrue(panel.moon_info.text().startswith('Auto:'))
        # The aberration boxes are saved with the params.
        panel._aberration_boxes['coma'].setChecked(False)
        self.assertEqual([panel.get_params()[k] for k in ('astigmatism', 'coma', 'trefoil')],
                         [True, False, True])

    def test_unticking_auto_keeps_its_moon(self):
        panel, _ = self.panel()
        panel.auto.setChecked(False)
        params = panel.get_params()
        self.assertEqual((round(params['moon_x']), round(params['moon_y'])), MOON)

    def test_tools_menu_runs_with_canvas_picking(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window._set_document(ImageDocument(np.repeat(scene()[..., None], 3, axis=2)))
        tools = [a for a in window.menuBar().actions() if a.text() == '&Tools'][0].menu()
        self.assertIn(window._optical_act, tools.actions())
        seen = []

        def drive():
            panel = window._active_filter_dlg
            try:
                seen.append((window._canvas._pick_enabled,
                             any(m.isVisible() for m in window._canvas._markers)))
                window._canvas.region_picked.emit(FAINT_MOON[0], FAINT_MOON[1], 0, 0)
                seen.append(panel.get_params()['auto'])
                panel.preview.setChecked(False)
                panel._accept()
            except BaseException as exc:
                seen.append(exc)
                panel._reject()
        QTimer.singleShot(0, drive)
        window._run_optical_correction()
        self.assertEqual(seen, [(True, True), False])
        self.assertFalse(window._canvas._pick_enabled)
        self.assertFalse(any(m.isVisible() for m in window._canvas._markers))
        op = window._undo.stack.applied_operations()[-1]
        self.assertEqual(op.filter_id, 'optical_correction')
        self.assertEqual((round(op.params['moon_x']), round(op.params['moon_y'])), FAINT_MOON)
        window._document.modified = False


if __name__ == '__main__':
    unittest.main()
