package chk1g16_ty1g16;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

public abstract class Run{

    protected abstract void run();
    protected abstract LiblinearAnnotator<FImage, String> trainAnnotator();


    //evaluate the performance of the model, training on 80% of the data and testing on the other 20%
    protected void testPerformance(){
        GroupedRandomSplitter<String, FImage> split_data =
                new GroupedRandomSplitter<String, FImage>(
                        Main.train_data, 80, 0, 20);
        Main.train_data = split_data.getTrainingDataset();

        LiblinearAnnotator<FImage, String> annotator = trainAnnotator();

        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        annotator, split_data.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getSummaryReport());
        System.out.println();
        System.out.println(result.getDetailReport());
    }

    protected void saveGuesses(String predictionsFileName, LiblinearAnnotator<FImage, String> annotator){
        File file = new File(predictionsFileName);

        try (BufferedWriter buffWriter = new BufferedWriter(new FileWriter(file.getAbsoluteFile()))) {
            System.out.println("Saving predictions in: " + file.getName() + "  ...");
            if (!file.exists()) {
                file.createNewFile();
            }

            boolean first = true;
            for (Integer id : Main.testImageToName.keySet()) {
                // Save file name and predicted class
                String predictedClass = annotator.classify(Main.testImageToName.get(id)).getPredictedClasses().toString();
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