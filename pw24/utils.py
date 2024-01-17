from quality_assessment.measures.full_reference import *
from quality_assessment.measures.no_reference import GeneralisedSignalToNoiseRatio

MEASURES = [UniversalQualityIndex(),
            StructuralSimilarityIndex(),
            MutualInformation(),
            # Quantitative Measures
            RootMeanSquaredError(),
            CosineDistance(),
            # Distribution-based measures
            WassersteinDistance(),
            JensenShannonDivergence(),
            KullbackLeiblerDivergence(),
            BhattacharyyaDistance()
            ]

STUBS = [
    "UQI", "SSIM", "MI", "RSME", "CD", "WD", "JSD", "KLD", "BD"
]

def calc_results(reference, reconstruction):
    results = []
    for measure in MEASURES:
        results.append(measure.compute_measure(reference, reconstruction))
    return results


def create_image_view(axis, results, all_results):

    maxima = np.max(all_results, axis=0)
    minima = np.min(all_results, axis=0)
    results = 1 - ((results - minima) / (maxima - minima))

    axis.set_xlim(-1, len(results))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for res_idx, res in enumerate(results):
        axis.bar(res_idx, res)
        axis.text(res_idx-0.2, 0.1, STUBS[res_idx], rotation=90)
