package chk1g16_ty1g16;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
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
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.awt.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class App {
    private static final String TRAIN_DATA = "http://comp3204.ecs.soton.ac.uk/cw/training.zip";
    private static final String TEST_DATA = "http://comp3204.ecs.soton.ac.uk/cw/testing.zip";

    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> data = new VFSGroupDataset<FImage>(TRAIN_DATA, ImageUtilities.FIMAGE_READER);
        //split into test/train but use the test bit for validation since OpenIMAJ does not allow train/val split
        GroupedRandomSplitter<String, FImage> split_data =
                new GroupedRandomSplitter<String, FImage>(data, 80, 0, 20);
        //for every class use 80 images for training and 20 for validation
        GroupedDataset<String, ListDataset<FImage>, FImage> train_data = split_data.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> val_data = split_data.getTestDataset();

        int width = 8;
        int height = 8;
        int stepX = 4;
        int stepY = 4;
        PixelSampler pixelSampler = new PixelSampler(stepX, stepY,width, height);

        HardAssigner<float[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(
                        train_data, 30), pixelSampler);

        FeatureExtractor<DoubleFV, FImage> extractor = new Extractor(assigner, pixelSampler);
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTILABEL,
                SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(train_data);



        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        ann, val_data, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println();
        System.out.println(result.getSummaryReport());
        System.out.println();
        System.out.println(result.getDetailReport());
    }

    static class Extractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;
        PixelSampler pixelSampler;

        public Extractor(HardAssigner<float[], float[], IntFloatPair> assigner, PixelSampler pixelSampler){
            this.assigner = assigner;
            this.pixelSampler = pixelSampler;
        }

        public DoubleFV extractFeature(FImage image) {
//            pdsift.analyseImage(image);

            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

            BlockSpatialAggregator<float[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<float[], SparseIntFV>(
                                bovw, 2, 2);



            return spatial.aggregate(pixelSampler.analyseImage(image), image.getBounds())
                    .normaliseFV();
        }
    }

    static class PixelSampler {
        int stepX;
        int stepY;
        int width;
        int height;

        public PixelSampler (int stepX, int stepY, int width, int height){
            this.stepX = stepX;
            this.stepY = stepY;
            this.width = width;
            this.height = height;
        }

        public LocalFeatureList<FloatKeypoint> analyseImage(FImage img){
            LocalFeatureList<FloatKeypoint> featList = new MemoryLocalFeatureList();
            RectangleSampler rectSampler =
                    new RectangleSampler(img, stepX,stepY, width, height);
            Iterator<FImage> imgIter = rectSampler.subImageIterator(img);
            Iterator<Rectangle> rectIter = rectSampler.iterator();

            System.out.println("bounds: " + img.getBounds());
            while(imgIter.hasNext()){
                Rectangle currRect = rectIter.next();
                FImage patch = imgIter.next();
                System.out.println("x " + currRect.x + " y: " + currRect.y);

                float[] featureVector = new float[patch.pixels.length * patch.pixels[0].length];

                int index = 0;
                for (int r = 0; r < patch.pixels.length; r++ ) {
                    for (int c = 0; c < patch.pixels[r].length; c++ ) {
                        featureVector[index] = patch.pixels[r][c];
                        index++;
                    }
                }

                featList.add(new FloatKeypoint(currRect.x, currRect.y,1,1,featureVector));
            }
            System.exit(0);
            return featList;
        }
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> dataset, PixelSampler pixelSampler)
    {
//        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys =
//                new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        List<LocalFeatureList<FloatKeypoint>> allKeys =
                new ArrayList<LocalFeatureList<FloatKeypoint>>();


        for (FImage img : dataset) {
            //System.out.println(img.getBounds());
            allKeys.add(pixelSampler.analyseImage(img));
            //System.exit(0);
        }


//        if (allFeatureVectors.size() > 10000) {
//            allFeatureVectors = allFeatureVectors.subList(0, 10000);
//        }
//
//        for (FImage rec : dataset) {
//            FImage img = rec.getImage();
//
//            pdsift.analyseImage(img);
//            allkeys.add(pdsift.getByteKeypoints(0.005f));
//        }
//
        if (allKeys.size() > 10000) {
            //allKeys = allKeys.subList(0, 10000);
        }

        DataSource<float[]> datasource =
                new LocalFeatureListDataSource<FloatKeypoint, float[]>(allKeys);

        //use 500 clusters
        FloatKMeans km = FloatKMeans.createExact(500);
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }








}
