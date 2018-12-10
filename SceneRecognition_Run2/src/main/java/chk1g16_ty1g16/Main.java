package chk1g16_ty1g16;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {
    private static final String TRAIN_DATA_DIR = "C:\\Users\\chkar\\Desktop\\CV data\\training";
    private static final String TEST_DATA_DIR = "C:\\Users\\chkar\\Desktop\\CV data\\testing";

    protected static GroupedDataset<String, ListDataset<FImage>, FImage> train_data;
    protected static Map<Integer, FImage> testIDToImage;


    public static void main(String[] args) throws FileSystemException {
        loadData();
        Run1 r = new Run1();
        r.savePredictions();
    }



    private static void loadData() throws FileSystemException {
        //load training data
        GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset = new VFSGroupDataset<FImage>(
                TRAIN_DATA_DIR,
                ImageUtilities.FIMAGE_READER);

        train_data = new MapBackedDataset<String, ListDataset<FImage>, FImage>();
        for (String groupName : dataset.getGroups()) {
            train_data.put(groupName, dataset.get(groupName));
        }

        //load test data
        File folder = new File(TEST_DATA_DIR);
        File[] listOfFiles = folder.listFiles();

        testIDToImage = new TreeMap();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().endsWith(".jpg")) {
                try {
                    FImage fImage = ImageUtilities.readF(listOfFiles[i]);
                    testIDToImage.put(Integer.parseInt(listOfFiles[i].getName().split("\\.")[0]) , fImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
