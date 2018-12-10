package chk1g16_ty1g16;

import com.google.common.collect.*;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import java.util.*;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.processing.resize.ResizeProcessor;

/*
Performs image classification using the KNN algorithm based on the "tiny image" feature.
 */
public class Run1 extends Run {

    private final int K = 4; // k for the KNN algorithm
    private final int tinyImageSize = 16; // width and height for resized tiny images
    private final String predictionsFilePath = "run1.txt";
    private HashMap<String, ArrayList<DoubleFV>> trainClassNameToFVs;


    public Run1() throws NullPointerException {
        if (Main.train_data == null) {
            throw new NullPointerException("No training data has been loaded!");
        }
        this.trainClassNameToFVs = getTinyImgFVs(Main.train_data);
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


    //classify one image using KNN algorithm
    private String K_NN_Classify(FImage imgToClassify) {
        DoubleFV imgToClassifyFV = getTinyImageFeatureVector(imgToClassify);

        //ordered in descending order
        //allows for multiple classes with the same distance
        TreeMultimap<Double, String> distanceToClasses =
                                TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());
        for (String trainClassName : trainClassNameToFVs.keySet()) {
            for (DoubleFV featVector : trainClassNameToFVs.get(trainClassName)) {
                //don't compare with itself in case the image is part of the training set
                if (featVector.equals(imgToClassifyFV)) {
                    continue;
                }

                // Calculate the euclidean distance
                double distance = featVector.compare(imgToClassifyFV, DoubleFVComparison.EUCLIDEAN);
                // Only keep 3 decimal places
                distance = Math.round(distance * 1000d) / 1000d;
                distanceToClasses.put(distance, trainClassName);
            }
        }

        //count number of neighbours belonging to each class
        int i = 0;
        HashMap<String, Integer> classNameToCount = new HashMap<>();
        outerLoop:
        for (Double distance : distanceToClasses.keySet()) {
            for (String className : distanceToClasses.get(distance)) {
                if (i == K) {
                    break outerLoop;
                }
                if (classNameToCount.containsKey(className)) {
                    classNameToCount.put(className, classNameToCount.get(className) + 1);
                } else {
                    classNameToCount.put(className, 1);
                }
                i++;
            }
        }
        //find the class with the most representatives in the neighbor set
        String prediction = null;
        int predictionCount = 0;
        for (String className : classNameToCount.keySet()) {
            if (classNameToCount.get(className) > predictionCount) {
                prediction = className;
                predictionCount = classNameToCount.get(className);
            }
        }

        return prediction;
    }

    //turn all images to tiny images cropped around the centre and extract their feature vectors
    private HashMap<String, ArrayList<DoubleFV>> getTinyImgFVs(GroupedDataset<String,
            ListDataset<FImage>, FImage> trainingSet) {
        HashMap<String, ArrayList<DoubleFV>> classNameToFVs =
                                           new HashMap<String, ArrayList<DoubleFV>>();
        
        for (String className : trainingSet.keySet()) {
            ArrayList<DoubleFV> trainFeatVector = new ArrayList<DoubleFV>();

            for (FImage trainImage : trainingSet.get(className)) {
                trainFeatVector.add(getTinyImageFeatureVector(trainImage));
                classNameToFVs.put(className, trainFeatVector);
            }
        }
        
        return classNameToFVs;
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
