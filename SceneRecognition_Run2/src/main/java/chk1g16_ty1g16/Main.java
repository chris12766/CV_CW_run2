package chk1g16_ty1g16;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Main {
    protected static final String TRAIN_DATA_DIR = "C:\\Users\\chkar\\Desktop\\CV data\\training";
    protected static final String TEST_DATA_DIR = "C:\\Users\\chkar\\Desktop\\CV data\\testing";
    protected static final String FVQ_CACHE = "run2_FVQ";

    protected static GroupedDataset<String, VFSListDataset<FImage>, FImage> train_data;
    protected static Map<Integer, FImage> testImageToName;


    public static void main(String[] args) throws FileSystemException {
        loadData();

        //Run run1 = new Run1();
        //run1.run();
        Run run2 = new Run2();
        run2.run();
    }

    protected static abstract class Run{
        protected abstract void run();
    }

    private static void loadData() throws FileSystemException {
        train_data = new VFSGroupDataset<FImage>(
                TRAIN_DATA_DIR,
                ImageUtilities.FIMAGE_READER);

        File folder = new File(TEST_DATA_DIR);
        File[] listOfFiles = folder.listFiles();

        testImageToName = new TreeMap();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().endsWith(".jpg")) {
                try {
                    FImage fImage = ImageUtilities.readF(listOfFiles[i]);
                    testImageToName.put(Integer.parseInt(listOfFiles[i].getName().split("\\.")[0]) , fImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    protected static void saveGuesses(Run run, LiblinearAnnotator<FImage, String> annotator){
        String predictionsFileName;
        if (run instanceof Run1) {
            predictionsFileName = "run1.txt";
        } else {
            predictionsFileName = "run2.txt";
        }

        File file = new File(predictionsFileName);

        try (BufferedWriter buffWriter = new BufferedWriter(new FileWriter(file.getAbsoluteFile()))) {
            System.out.println("Saving predictions in: " + file.getName() + "  ...");
            if (!file.exists()) {
                file.createNewFile();
            }

            boolean first = true;
            for (Integer id : testImageToName.keySet()) {
                // Save file name and predicted class
                String predictedClass = annotator.classify(testImageToName.get(id)).getPredictedClasses().toString();
                if (!first) {
                    buffWriter.write("\n");
                } else {
                    first = false;
                }
                buffWriter.write(id + ".jpg " + predictedClass.substring(1, predictedClass.length() - 1));
            }
            System.out.println("Saving of predictions done.");
        } catch (IOException e) {
            System.err.println("Saving of predictions failed.");
            e.printStackTrace();
        }
    }

}
