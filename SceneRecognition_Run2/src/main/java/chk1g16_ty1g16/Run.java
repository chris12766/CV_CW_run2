package chk1g16_ty1g16;

import org.openimaj.image.FImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public abstract class Run{

    //classify one 1 image
    protected abstract String predict(FImage image);
    protected abstract String getPredictionsFilePath();

    //predict on the test data and save the predictions to txt file
    protected void savePredictions(){
        File file = new File(getPredictionsFilePath());

        try (BufferedWriter buffWriter = new BufferedWriter(new FileWriter(file.getAbsoluteFile()))) {
            System.out.println("Saving predictions in: " + file.getName() + "  ...");
            if (!file.exists()) {
                file.createNewFile();
            }

            boolean first = true;
            for (Integer id : Main.testIDToImage.keySet()) {
                // Save file name and predicted class
                String predictedClass = predict(Main.testIDToImage.get(id));

                //cut out irrelevant parts
                if (getPredictionsFilePath().equals("run2.txt")) {
                    predictedClass = predictedClass.substring(1, predictedClass.length() - 1);
                }

                if (!first) {
                    buffWriter.write("\n");
                } else {
                    first = false;
                }
                buffWriter.write(id + ".jpg " + predictedClass);
            }
            System.out.println("Saving of predictions done.");
        } catch (IOException e) {
            System.err.println("Saving of predictions failed.");
            e.printStackTrace();
        }
    }
}