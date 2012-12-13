// package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;

public class NER {

    public static void main(String[] args) throws IOException {
        if (args.length < 5) {
            System.out
                    .println("USAGE: java -cp classes NER ./data/train ./data/dev" 
                            + " [Regularization] [Iterations] [Rate]");
            return;
        }

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> testData = FeatureFactory.readTestData(args[1]);

        System.err.println("Corpus loaded.");
        // read the train and test data
        // TODO: Implement this function (just reads in vocab and word vectors)
        FeatureFactory.initializeVocab("./data/vocab.txt");
        SimpleMatrix allVecs = FeatureFactory
                .readWordVectors("./data/wordVectors.txt");

        System.err.println("Vocabulary loaded.");

        // initialize model
        WindowModel model = new WindowModel(5,
                100, 
                Double.parseDouble(args[2]),
                Integer.parseInt(args[3]),
                Double.parseDouble(args[4]));

        model.initWeights();

        // TODO: Implement those two functions
        model.train(trainData);
        model.test(testData);
    }
}
