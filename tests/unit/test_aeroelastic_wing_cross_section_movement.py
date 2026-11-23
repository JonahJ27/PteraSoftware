"""This module contains classes to test AeroelasticWingCrossSectionMovements."""

import unittest

import numpy as np
import numpy.testing as npt

import pterasoftware as ps
from tests.unit.fixtures import (
    aeroelastic_wing_cross_section_movement_fixtures,
    geometry_fixtures,
)


class TestAeroelasticWingCrossSectionMovement(unittest.TestCase):
    """This is a class with functions to test AeroelasticWingCrossSectionMovements."""

    def test_is_subclass_of_core(self) -> None:
        """Test that AeroelasticWingCrossSectionMovement is a subclass of
        CoreWingCrossSectionMovement."""
        self.assertTrue(
            issubclass(
                ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement,
                ps._core.CoreWingCrossSectionMovement,
            )
        )

    def test_instantiation_returns_correct_type(self) -> None:
        """Test that AeroelasticWingCrossSectionMovement instantiation returns an
        AeroelasticWingCrossSectionMovement."""
        base_wing_cross_section = (
            geometry_fixtures.make_root_wing_cross_section_fixture()
        )
        aeroelastic_wing_cross_section_movement = ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement(
            base_wing_cross_section=base_wing_cross_section,
        )
        self.assertIsInstance(
            aeroelastic_wing_cross_section_movement,
            ps.movements.aeroelastic_wing_cross_section_movement.AeroelasticWingCrossSectionMovement,
        )

    def test_generate_wing_cross_sections_returns_wing_cross_sections(self) -> None:
        """Test that generate_wing_cross_sections returns WingCrossSections when called
        through the public class."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )
        wing_cross_sections = (
            aeroelastic_wing_cross_section_movement.generate_wing_cross_sections(
                num_steps=5, delta_time=0.01
            )
        )
        self.assertEqual(len(wing_cross_sections), 5)
        for wing_cross_section in wing_cross_sections:
            self.assertIsInstance(
                wing_cross_section,
                ps.geometry.wing_cross_section.WingCrossSection,
            )

    def test_generate_wing_cross_section_at_time_step_returns_wing_cross_section(
        self,
    ) -> None:
        """Test that generate_wing_cross_section_at_time_step returns a
        WingCrossSection."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )
        wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=2, delta_time=0.01
        )
        self.assertIsInstance(
            wing_cross_section,
            ps.geometry.wing_cross_section.WingCrossSection,
        )

    def test_no_deformation_matches_static_base(self) -> None:
        """Test that, with no prescribed movement and no deformation, the generated
        WingCrossSection matches the base WingCrossSection."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_static_aeroelastic_wing_cross_section_movement_fixture()
        )
        base_wing_cross_section = (
            aeroelastic_wing_cross_section_movement.base_wing_cross_section
        )

        wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=3, delta_time=0.01
        )

        npt.assert_allclose(
            wing_cross_section.Lp_Wcsp_Lpp,
            base_wing_cross_section.Lp_Wcsp_Lpp,
            rtol=1e-10,
            atol=1e-14,
        )
        npt.assert_allclose(
            wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            base_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_default_deformation_matches_explicit_none(self) -> None:
        """Test that omitting deformationAngles_Wcsp_to_Wcs_ixyz produces the same
        result as explicitly passing None."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )

        default_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=2, delta_time=0.01
        )
        explicit_none_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=2, delta_time=0.01, deformationAngles_Wcsp_to_Wcs_ixyz=None
        )

        npt.assert_array_equal(
            default_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            explicit_none_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
        )
        npt.assert_array_equal(
            default_wing_cross_section.Lp_Wcsp_Lpp,
            explicit_none_wing_cross_section.Lp_Wcsp_Lpp,
        )

    def test_deformation_adds_to_prescribed_angles(self) -> None:
        """Test that deformationAngles_Wcsp_to_Wcs_ixyz is added to the prescribed
        angles_Wcsp_to_Wcs_ixyz."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_static_aeroelastic_wing_cross_section_movement_fixture()
        )
        base_wing_cross_section = (
            aeroelastic_wing_cross_section_movement.base_wing_cross_section
        )
        deformationAngles_Wcsp_to_Wcs_ixyz = np.array([5.0, 5.0, 5.0], dtype=float)

        wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=1,
            delta_time=0.01,
            deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
        )

        # With static movement, the prescribed angles equal the base angles, so the
        # result should be the base angles plus the deformation.
        expected_angles = (
            base_wing_cross_section.angles_Wcsp_to_Wcs_ixyz
            + deformationAngles_Wcsp_to_Wcs_ixyz
        )
        npt.assert_allclose(
            wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            expected_angles,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_deformation_leaves_Lp_unchanged(self) -> None:
        """Test that deformationAngles_Wcsp_to_Wcs_ixyz does not affect the generated
        Lp_Wcsp_Lpp."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_static_aeroelastic_wing_cross_section_movement_fixture()
        )
        base_wing_cross_section = (
            aeroelastic_wing_cross_section_movement.base_wing_cross_section
        )
        deformationAngles_Wcsp_to_Wcs_ixyz = np.array([5.0, 5.0, 5.0], dtype=float)

        wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=1,
            delta_time=0.01,
            deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
        )

        npt.assert_allclose(
            wing_cross_section.Lp_Wcsp_Lpp,
            base_wing_cross_section.Lp_Wcsp_Lpp,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_zero_deformation_matches_no_deformation(self) -> None:
        """Test that a zero deformation produces the same result as no deformation."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )

        no_deformation_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=2, delta_time=0.01
        )
        zero_deformation_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=2,
            delta_time=0.01,
            deformationAngles_Wcsp_to_Wcs_ixyz=np.zeros(3, dtype=float),
        )

        npt.assert_allclose(
            zero_deformation_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            no_deformation_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            rtol=1e-10,
            atol=1e-14,
        )

    def test_deformation_adds_on_top_of_oscillation(self) -> None:
        """Test that deformation is added on top of the oscillating prescribed
        angles."""
        aeroelastic_wing_cross_section_movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )
        deformationAngles_Wcsp_to_Wcs_ixyz = np.array([3.0, -2.0, 1.0], dtype=float)

        prescribed_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=3, delta_time=0.01
        )
        deformed_wing_cross_section = aeroelastic_wing_cross_section_movement.generate_wing_cross_section_at_time_step(
            step=3,
            delta_time=0.01,
            deformationAngles_Wcsp_to_Wcs_ixyz=deformationAngles_Wcsp_to_Wcs_ixyz,
        )

        # The deformed angles should differ from the prescribed angles by exactly the
        # deformation.
        npt.assert_allclose(
            deformed_wing_cross_section.angles_Wcsp_to_Wcs_ixyz
            - prescribed_wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
            deformationAngles_Wcsp_to_Wcs_ixyz,
            rtol=1e-10,
            atol=1e-14,
        )


class TestAeroelasticWingCrossSectionMovementStepParameters(unittest.TestCase):
    """This is a class with functions to test the per time step parameters of
    AeroelasticWingCrossSectionMovements."""

    def test_parameters_match_the_built_wing_cross_section(self) -> None:
        """Test that the parameters, with and without deformation, equal the values of
        the WingCrossSection built at the same time step with the same deformation, bit
        for bit, with read only arrays."""
        movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )
        for deformation in (None, np.array([0.1, -0.25, 0.3], dtype=float)):
            for step in (0, 1, 7, 23, 61):
                with self.subTest(deformation=deformation, step=step):
                    parameters = (
                        movement._generate_wing_cross_section_parameters_at_time_step(
                            step,
                            0.01,
                            deformationAngles_Wcsp_to_Wcs_ixyz=deformation,
                        )
                    )
                    wing_cross_section = (
                        movement.generate_wing_cross_section_at_time_step(
                            step,
                            0.01,
                            deformationAngles_Wcsp_to_Wcs_ixyz=deformation,
                        )
                    )
                    self.assertTrue(
                        np.array_equal(
                            parameters.Lp_Wcsp_Lpp, wing_cross_section.Lp_Wcsp_Lpp
                        )
                    )
                    self.assertTrue(
                        np.array_equal(
                            parameters.angles_Wcsp_to_Wcs_ixyz,
                            wing_cross_section.angles_Wcsp_to_Wcs_ixyz,
                        )
                    )
                    self.assertFalse(parameters.angles_Wcsp_to_Wcs_ixyz.flags.writeable)

    def test_deformation_is_added_to_the_prescribed_angles(self) -> None:
        """Test that the deformation is added to the prescribed angles."""
        movement = (
            aeroelastic_wing_cross_section_movement_fixtures.make_basic_aeroelastic_wing_cross_section_movement_fixture()
        )
        deformation = np.array([0.1, -0.25, 0.3], dtype=float)
        prescribed = movement._generate_wing_cross_section_parameters_at_time_step(
            7, 0.01
        )
        deformed = movement._generate_wing_cross_section_parameters_at_time_step(
            7, 0.01, deformationAngles_Wcsp_to_Wcs_ixyz=deformation
        )
        self.assertTrue(
            np.array_equal(
                deformed.angles_Wcsp_to_Wcs_ixyz,
                prescribed.angles_Wcsp_to_Wcs_ixyz + deformation,
            )
        )
        self.assertTrue(np.array_equal(deformed.Lp_Wcsp_Lpp, prescribed.Lp_Wcsp_Lpp))
