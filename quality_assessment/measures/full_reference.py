from quality_assessment.measures import FullReferenceMeasure
from sewar import ssim, uqi, rmse
from sklearn.metrics import mutual_info_score


class StructuralSimilarityIndex(FullReferenceMeasure):

    def compute_measure(self, ground_truth_image, reconstructed_image):
        return ssim(ground_truth_image, reconstructed_image, ws=8)[0]


class UniversalQualityIndex(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return uqi(expected_result, reconstructed_image, ws=8)


class MeanSquaredError(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return rmse(expected_result, reconstructed_image)


class MutualInformation(FullReferenceMeasure):

    def compute_measure(self, expected_result, reconstructed_image):
        return mutual_info_score(expected_result.reshape((-1, )), reconstructed_image.reshape((-1, )))
