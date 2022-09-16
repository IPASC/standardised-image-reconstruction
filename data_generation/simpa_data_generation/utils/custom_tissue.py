import simpa as sp


def constant(mua=1e-10, mus=1e-10, g=1e-10, segmentation_class=None):
    """
    Get a molecular composition that is constant over wavelength
    """
    if segmentation_class is None:
        segmentation_class = sp.SegmentationClasses.GENERIC

    return (sp.MolecularCompositionGenerator().append(sp.Molecule(name="constant_mua_mus_g",
                                                            absorption_spectrum=
                                                            sp.AbsorptionSpectrumLibrary().CONSTANT_ABSORBER_ARBITRARY(
                                                                mua),
                                                            volume_fraction=1.0,
                                                            scattering_spectrum=
                                                            sp.ScatteringSpectrumLibrary.
                                                            CONSTANT_SCATTERING_ARBITRARY(mus),
                                                            anisotropy_spectrum=
                                                            sp.AnisotropySpectrumLibrary.
                                                            CONSTANT_ANISOTROPY_ARBITRARY(g)))
            .get_molecular_composition(segmentation_class))