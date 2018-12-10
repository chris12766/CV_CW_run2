package chk1g16_ty1g16;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.lang.ArrayUtils;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.io.IOUtils;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/*
Performs scene classification using a set of one-vs-all linear classifiers
based on the bag-of-visual-words feature.
 */
public class Run2 extends Run{

    private final String fvqCache = "run2_FVQ";
    private LiblinearAnnotator<FImage, String> imageAnnotator;
    private final String predictionsFilePath = "run2.txt";

    //classify an image
    @Override
    protected String predict(FImage image) {
        return getTrainedAnnotator().classify(image).getPredictedClasses().toString();
    }

    @Override
    protected String getPredictionsFilePath() {
        return predictionsFilePath;
    }

    //evaluate the performance of the model, training on 80% of the labelled data and testing on the other 20%
    protected void evaluatePerformance() {
        GroupedRandomSplitter<String, FImage> split_data =
                new GroupedRandomSplitter<String, FImage>(
                        Main.train_data, 80, 0, 20);
        Main.train_data = split_data.getTrainingDataset();

        LiblinearAnnotator<FImage, String> annotator = getTrainedAnnotator();

        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        annotator, split_data.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getSummaryReport());
        System.out.println();
        System.out.println(result.getDetailReport());
    }

    //train annotator on all available labelled data, or use a pre-trained one
    private LiblinearAnnotator<FImage, String> getTrainedAnnotator() throws NullPointerException {
        if(imageAnnotator == null) {
            if (Main.train_data == null) {
                throw new NullPointerException("No training data has been loaded!");
            }
            GroupedDataset<String, ListDataset<FImage>, FImage> test_data = null;
            //sampling step in x and y directions
            int step = 8;
            //each sampled patch is size x size
            int size = 4;
            DensePixelPatchSampler pixelSampler = new DensePixelPatchSampler(step, size);
            //construct a feature vector quantiser to map pixel patches to visual words
            HardAssigner<float[], float[], IntFloatPair> fvq =
                    getFeatureVectorQuantiser(Main.train_data, pixelSampler, 30);
            //computes a spatial histogram for each image
            FeatureExtractor<DoubleFV, FImage> extractor = new DenseFeatureExtractor(fvq, pixelSampler);
            //construct and train one-vs-all linear classifiers (1 per image class)
            LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<FImage, String>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS,
                    SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
            annotator.train(Main.train_data);
            imageAnnotator = annotator;
        }
        return imageAnnotator;
    }

    //creates spatial bag-of-visual-words histograms for images
    private class DenseFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> fvq;
        DensePixelPatchSampler pixelSampler;

        public DenseFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> fvq,
                                     DensePixelPatchSampler pixelSampler){
            this.fvq = fvq;
            this.pixelSampler = pixelSampler;
        }

        public DoubleFV extractFeature(FImage image) {
            //map each feature to a visual word
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(fvq);


            //construct spatial histograms
            BlockSpatialAggregator<float[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<float[], SparseIntFV>(
                                bovw, 2, 4);


            //append and normalise all spatial histograms for the image
            return spatial.aggregate(pixelSampler.getFloatKeypoints(image), image.getBounds())
                    .normaliseFV();
        }
    }

    //extracts the feature vectors of pixel patches that form a dense grid across the image
    private class DensePixelPatchSampler {
        private int step;
        private int size;


        public DensePixelPatchSampler (int step, int size){
            this.step = step;
            this.size = size;
        }

        public LocalFeatureList<FloatKeypoint> getFloatKeypoints(FImage img){
            LocalFeatureList<FloatKeypoint> featList = new MemoryLocalFeatureList();
            RectangleSampler rectSampler =
                    new RectangleSampler(img, step, step, size, size);

            //extract feature vectors from densely sampled patches from the image
            for (Rectangle patch : rectSampler.allRectangles()){
                FImage normalisedSubimage = img.extractROI(patch).normalise();
                float[] featureVector = getFeatureVector(normalisedSubimage);
                featList.add(new FloatKeypoint(patch.x, patch.y, 0, 1, featureVector));
            }

            return featList;
        }

        // flatten subimage pixels into a feature vector
        private float[] getFeatureVector(FImage img) {
            float[] featureVector = new float[img.height*img.width];

            for (int r = 0; r < img.height; r++ ) {
                featureVector = ArrayUtils.addAll(featureVector, img.pixels[r]);
            }
            return featureVector;
        }
    }

    //uses K-Means to map feature vectors to visual words
    private HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset, DensePixelPatchSampler pixelSampler)
    {
        List<LocalFeatureList<FloatKeypoint>> allKeys = new ArrayList<LocalFeatureList<FloatKeypoint>>();

        int imagesProcessed = 0;
        for (FImage img : groupedDataset) {
            allKeys.add(pixelSampler.getFloatKeypoints(img));
            imagesProcessed++;
        }

        DataSource<float[]> datasource =
                new LocalFeatureListDataSource<FloatKeypoint, float[]>(allKeys);

        //use 500 clusters for the K-Means algorithm
        FloatKMeans km = FloatKMeans.createExact(500);
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    //use cache FVQ if available, otherwise train a new one and save it
    private HardAssigner<float[], float[], IntFloatPair> getFeatureVectorQuantiser(
            GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset,
            DensePixelPatchSampler sampler, int numTrainSamples) {
        HardAssigner<float[], float[], IntFloatPair> fvq = null;
        File fvqFile = new File(fvqCache);

        System.out.println("Loading a Feature Vector Quantiser from: " + fvqCache + "  ...");
        if(fvqFile.exists()) {
            try {
                fvq = IOUtils.readFromFile(fvqFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if(fvq == null) {
            System.out.println("Loading failed.");
            System.out.println("Training new Feature Vector Quantiser ...");
            fvq = trainQuantiser(GroupedUniformRandomisedSampler.sample(groupedDataset, numTrainSamples), sampler);
            System.out.println("Training completed.");
            try {
                System.out.println("Saving the Feature Vector Quantiser to: " + fvqCache + "  ...");
                IOUtils.writeToFile(fvq, fvqFile);
                System.out.println("Saving completed.");
            } catch (IOException e) {
                System.out.println("Saving failed.");
                e.printStackTrace();
            }
        }
        return fvq;
    }

}
