package chk1g16_ty1g16;

import com.google.common.collect.*;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import java.util.*;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.ObjectNearestNeighboursExact;
import org.openimaj.util.pair.IntFloatPair;

/*
Performs image classification using the KNN algorithm based on the "tiny image" feature.
 */
public class Run1 extends Run {

    private int K = 3; // k for the KNN algorithm
    private final int tinyImageSize = 16; // width and height for resized tiny images
    private final String predictionsFilePath = "run1.txt";
    private HashMap<Integer, String> imageIndextoClassName;
    private List<DoubleFV> trainDataFVs;


    public Run1() throws NullPointerException {
        if (Main.train_data == null) {
            throw new NullPointerException("No training data has been loaded!");
        }
        this.imageIndextoClassName = new HashMap<>();
        this.trainDataFVs = getTinyImgFVs(Main.train_data);
    }

    //classify one 1 image
    @Override
    protected String predict(FImage image) {
        return K_NN_Classify(image);
    }

    @Override
    protected String getPredictionsFilePath() {
        return predictionsFilePath;
    }


    //classify one image using the KNN algorithm
    private String K_NN_Classify(FImage imgToClassify) {
        DoubleFV imgToClassifyFV = getTinyImageFeatureVector(imgToClassify);

        //create the KNN classifier
        ObjectNearestNeighboursExact<DoubleFV> nnExact = new ObjectNearestNeighboursExact<>(trainDataFVs, DoubleFVComparison.EUCLIDEAN);
        //classify
        List<IntFloatPair> finalList = nnExact.searchKNN(imgToClassifyFV, K);

        return imageIndextoClassName.get(findMostCommonClassIndex(finalList));
    }

    //returns the index of the most common class in the list
    private int findMostCommonClassIndex(List<IntFloatPair> input) {
        HashMap<Integer, Integer> classIndexToCount = new HashMap<>();

        for (IntFloatPair a : input) {
            int classIndex = a.first;

            if (classIndexToCount.containsKey(classIndex)) {
                classIndexToCount.put(classIndex, classIndexToCount.get(classIndex) + 1);
            } else {
                classIndexToCount.put(classIndex, 1);
            }
        }

        //find the class with the most representatives in the neighbor set
        int predictionIndex = 0;
        int predictionCount = 0;
        for (Integer classIndex : classIndexToCount.keySet()) {
            if (classIndexToCount.get(classIndex) > predictionCount) {
                predictionIndex = classIndex;
                predictionCount = classIndexToCount.get(classIndex);
            }
        }

        return predictionIndex;
    }

    protected void evaluatePerformance() {
        // Split the dataset to 80% for training and 20% for testing
        GroupedRandomSplitter<String, FImage> split_data =
                new GroupedRandomSplitter<String, FImage>(Main.train_data, 80, 0, 20);
        GroupedDataset<String, ListDataset<FImage>, FImage> testData = split_data.getTestDataset();
        this.trainDataFVs = getTinyImgFVs(split_data.getTrainingDataset());

        double correct = 0;
        double count = 0;
        double bestAccuracy = 0;
        int bestK = 0;

        //experiment with different K's
        for (int i = 1; i < 20; i++){
            this.K = i;

            for (String label : testData.keySet()) {
                for (FImage fImage : testData.get(label)) {
                    String prediction = predict(fImage);
                    if (prediction.equals(label)) {
                        correct++;
                    }
                    count++;
                }
            }

            double accuracy = correct / count;
            System.out.println("Value of K is: " + K);
            System.out.println("Accuracy is: " + accuracy);
            System.out.println();

            if(bestAccuracy < accuracy){
                bestAccuracy = accuracy;
                bestK = K;
            }
        }

        System.out.println("The optimal K is: " + bestK);
        System.out.println("The best accuracy achieved is: " + bestAccuracy);
    }

    //turn all images to tiny images cropped around the centre and extract their feature vectors
    private ArrayList<DoubleFV> getTinyImgFVs(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet) {
        ArrayList<DoubleFV> featVectors = new ArrayList<DoubleFV>();
        int index = 0;
        for (String className : trainingSet.keySet()) {
            for (FImage trainImage : trainingSet.get(className)) {
                featVectors.add(getTinyImageFeatureVector(trainImage));
                imageIndextoClassName.put(index, className);
                index++;
            }
        }

        return featVectors;
    }

    //crop a tiny image around the centre of the passed image and extract its feature vector
    private DoubleFV getTinyImageFeatureVector(FImage rawImage) {
        FImage newImage;
        FImage tinyImage;

        //Get the width and height of image
        int width_Image = rawImage.getWidth();
        int height_Image = rawImage.getHeight();

        // Crop the image to square where the centre point is the original centre point
        if(width_Image > height_Image) {
            newImage = rawImage.extractCenter(height_Image, height_Image);
        }else if(width_Image < height_Image) {
            newImage = rawImage.extractCenter(width_Image, width_Image);
        }else {
            newImage = rawImage.extractCenter(width_Image, height_Image);
        }

        // Resize the image into tinyImageSize x tinyImageSize
        tinyImage = newImage.process(new ResizeProcessor(tinyImageSize,tinyImageSize));
        tinyImage.normalise();

        DoubleFV featureVector = new DoubleFV(tinyImage.getDoublePixelVector());

        return featureVector;
    }

}
