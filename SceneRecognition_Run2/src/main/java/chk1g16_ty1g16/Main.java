package chk1g16_ty1g16;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {
    private static final String TRAIN_DATA_URL = "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip";
    private static final String TEST_DATA_URL = "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip";

    protected static GroupedDataset<String, ListDataset<FImage>, FImage> train_data;
    protected static Map<Integer, FImage> testIDToImage;


    public static void main(String[] args) throws FileSystemException {
        loadData();

        //Run2 r2 = new Run2();
        //r2.evaluatePerformance();

        Run1 run1 = new Run1();
        run1.evaluatePerformance();
    }



    private static void loadData() throws FileSystemException {
        //load training data
        VFSGroupDataset<FImage> dataset =
                new VFSGroupDataset<FImage>(TRAIN_DATA_URL, ImageUtilities.FIMAGE_READER);

        train_data = new MapBackedDataset<String, ListDataset<FImage>, FImage>();
        for (String groupName : dataset.getGroups()) {
            train_data.put(groupName, dataset.get(groupName));
        }


        VFSListDataset<FImage> testData = new VFSListDataset<FImage>(TEST_DATA_URL, ImageUtilities.FIMAGE_READER);

        testIDToImage = new TreeMap();
        for(int i = 0; i < testData.size(); i++){
            String imagePath = testData.getFileObject(i).toString();

            int imageIndex = imagePath.lastIndexOf('/');
            String imageName = imagePath.substring(imageIndex+1);
            int imageID = Integer.parseInt(imageName.substring(0, imageName.length() - 4));
            testIDToImage.put(imageID, testData.get(i));
        }
    }

}
