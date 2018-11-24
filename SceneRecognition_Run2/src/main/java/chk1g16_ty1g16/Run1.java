package chk1g16_ty1g16;

import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

public class Run1 extends Run {

    //train and run on test data, saving the results
    public void run() {
        saveGuesses("run1.txt", trainAnnotator());
    }

    @Override
    protected LiblinearAnnotator<FImage, String> trainAnnotator() {
        return null;
    }
}
