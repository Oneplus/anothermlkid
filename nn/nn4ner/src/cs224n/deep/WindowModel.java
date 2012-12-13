// package cs224n.deep;

import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

    // 
    final static int MAX_ITERATION = 2;

    // model parameter
    protected SimpleMatrix L, W, U, b1;
    double b2;

    // C ----- regularization
    // m ----- total number of data
    // alpha - learning rate
    double C, m, alpha;
    //
    public int windowSize, wordSize, hiddenSize;
    public int iterations;

    public WindowModel(int _windowSize, int _hiddenSize, 
            double _C, int _iterations, double _alpha) {
        // TODO
        windowSize     = _windowSize;
        wordSize     = FeatureFactory.allVecs.numCols();
        hiddenSize     = _hiddenSize;
        C             = _C;
        iterations     = _iterations;
        alpha         = _alpha;
    }

    /**
     * Initializes the weights randomly.
     */
    public void initWeights() {
        // TODO
        // initialize with bias inside as the last column
        // W = SimpleMatrix...
        // U for the score
        // U = SimpleMatrix...
        int fanIn = windowSize * wordSize;
        int fanOut = hiddenSize;
        double epsilon_init = Math.sqrt(6. / (fanIn + fanOut));
        
        W = SimpleMatrix.random(fanOut, fanIn, -epsilon_init, epsilon_init, new Random());
        U = SimpleMatrix.random(fanOut, 1, -epsilon_init, epsilon_init, new Random());
        
        b1 = SimpleMatrix.random(fanOut, 1, -epsilon_init, epsilon_init, new Random());
        b2 = (2 * Math.random() - 1.) * epsilon_init;
        
        L = new SimpleMatrix(FeatureFactory.allVecs);
    }

    /**
     * Extract words in windows, mainly handle the start tag "<s>" and ending tag "</s>"
     * 
     *     @param i             instance position.
     *  @param _trainData     training data.
     *    @return words in windows
     */
    private String[] extractWordsInWindows(int i, List<Datum> _trainData) {
        String[] wordsInWindows = new String[windowSize];
        int centerOffset = windowSize / 2;
        wordsInWindows[centerOffset] = _trainData.get(i).word;

        boolean reachStart = false;
        for (int leftOffset = centerOffset - 1; leftOffset >= 0; leftOffset --) {
            if (i - (centerOffset - leftOffset) < 0
                    || _trainData.get(i - (centerOffset - leftOffset)).word.equals(".")) {
                reachStart = true;
            }

            if (reachStart) {
                wordsInWindows[leftOffset] = "<s>";
            } else {
                wordsInWindows[leftOffset] = _trainData.get(i - (centerOffset - leftOffset)).word;
            }
        }
        
        boolean reachEnd = false;
        for (int rightOffset = centerOffset + 1; rightOffset < windowSize; rightOffset ++) {
            if (i + (rightOffset - centerOffset) >= _trainData.size()
                    || _trainData.get(i + (rightOffset - centerOffset) - 1).word.equals(".")) {
                reachEnd = true;
            }
            
            if (reachEnd) {
                wordsInWindows[rightOffset] = "</s>";
            } else {
                wordsInWindows[rightOffset] = _trainData.get(i + rightOffset - centerOffset).word;
            }
        }
        
        return wordsInWindows;
    }

    /**
     * Calculate tanh of each element in matrix 
     * 
     * @param x
     * @return
     */
    private SimpleMatrix tanhMatrix(SimpleMatrix x) {
        int numRows = x.numRows();
        int numCols = x.numCols();
        SimpleMatrix z = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; ++ i) {
            for (int j = 0; j < numCols; ++ j) {
                z.set(i, j, Math.tanh(x.get(i, j)));
            }
        }
        return z;
    }

    /**
     * Calculate dtanhx=1-tanh^2x of each element in a matrix
     * 
     * @param x
     * @return
     */
    private SimpleMatrix tanhDetriveMatrix(SimpleMatrix x) {
        int numRows = x.numRows();
        int numCols = x.numCols();
        SimpleMatrix z = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; ++ i) {
            for (int j = 0; j < numCols; ++ j) {
                double value = Math.tanh(x.get(i, j));
                z.set(i, j, 1 - value * value);
            }
        }
        return z;
    }

    private double sigmoid(double x) {
        return 1. / (1. + Math.exp(-x));
    }

    private SimpleMatrix sigmoidMatrix(SimpleMatrix x) {
        int numRows = x.numRows();
        int numCols = x.numCols();
        
        SimpleMatrix z = new SimpleMatrix(numRows, numCols);
        
        for (int i = 0; i < numRows; ++ i) {
            for (int j = 0; j < numCols; ++ j) {
                z.set(i, j, sigmoid(x.get(i, j)));
            }
        }
        return z;
    }
    
    /**
     * Simplest SGD training
     */
    public void train(List<Datum> _trainData) {
        // TODO
        System.out.println("Start SGD training ...");
        System.out.println("");
        System.out.println("#(instances): " + _trainData.size());
        System.out.println("#(iterations): " + iterations);
        System.out.println("");
        
        m = _trainData.size();
        
        for (int iter_time = 0; iter_time < iterations; iter_time ++) {
            
            for (int i = 0; i < _trainData.size(); i ++) {
                if ((i + 1) % 10000 == 0)
                    System.out.println("Iter #" + (iter_time + 1) + " " + (i + 1) + " instances.");
                
                String[] wordsInWindows = extractWordsInWindows(i, _trainData);
                
                SimpleMatrix x = new SimpleMatrix(windowSize * wordSize, 1);
                for (int j = 0; j < windowSize; j ++) {
                    int index = FeatureFactory.wordToNum.containsKey(wordsInWindows[j]) ? FeatureFactory.wordToNum.get(wordsInWindows[j]) : 0;
                    x.insertIntoThis(wordSize * j, 0, L.extractVector(true, index).transpose());
                }
                
                double y = (_trainData.get(i).label).equals("O") ? 0. : 1.;

                // x \in R^(nC \times 1)
                // W \in R^(H \times nC)
                // z \in R^(H \times 1)
                SimpleMatrix z = tanhMatrix(W.mult(x).plus(b1));
                
                // dz \in R^(H \times 1)
                SimpleMatrix dz = tanhDetriveMatrix(W.mult(x).plus(b1));

                // h \in R^(1 \times 1)
                // h = sigmoid{ U^T tanh(WX + b1) + b2 }
                double h = sigmoid(U.transpose().mult(z).get(0, 0) + b2);
                
                // dJ/dh = -y/h + (1-y)/(1-h)
                double factor_h = -y * (1 - h) + (1 - y) * h;
                
                // dJ/dU = [-y*(1-h)+(1-y)*h]*z
                SimpleMatrix dJ_dU = z.scale(factor_h).plus(U.scale(C/m));
                
                // dJ/db2 = [-y*(1-h)+(1-y)*h]
                double dJ_db2 = h;
                
                // dJ/dW = [
                SimpleMatrix dJ_dW = dz.elementMult(U).mult(x.transpose()).scale(factor_h).plus(W.scale(C/m));
                
                // 
                SimpleMatrix dJ_db1 = dz.elementMult(U).scale(factor_h);

                // dJ_dL \in R^(nC \times 1)
                SimpleMatrix dJ_dL = W.transpose().mult(dz.elementMult(U)).scale(factor_h);

                U = U.minus(dJ_dU.scale(alpha));
                b2 = b2 - dJ_db2 * alpha;
                W = W.minus(dJ_dW.scale(alpha));
                b1 = b1.minus(dJ_db1.scale(alpha));

                for (int j = 0; j < windowSize; ++ j) {
                    int index = FeatureFactory.wordToNum.containsKey(wordsInWindows[j]) ? FeatureFactory.wordToNum.get(wordsInWindows[j]) : 0;
                    L.insertIntoThis(index, 0, dJ_dL.extractMatrix(j * wordSize, (j + 1) * wordSize, 0, 1).scale(alpha));
                }
            }
        }
        
        System.out.println("Training done.");
    }

    public void test(List<Datum> testData) {
        // TODO
        System.out.println("Start Testing ...");

        int correct = 0, gold = 0, pred = 0;
        for (int i = 0; i < testData.size(); ++ i) {
            String[] wordsInWindows = extractWordsInWindows(i, testData);
            
            SimpleMatrix x = new SimpleMatrix(windowSize * wordSize, 1);
            for (int j = 0; j < windowSize; j ++) {
                int index = FeatureFactory.wordToNum.containsKey(wordsInWindows[j]) ? FeatureFactory.wordToNum.get(wordsInWindows[j]) : 0;
                x.insertIntoThis(wordSize * j, 0, L.extractVector(true, index).transpose());
            }

            double h = sigmoid(U.transpose().mult(tanhMatrix(W.mult(x).plus(b1))).get(0, 0) + b2);
            
            int y = testData.get(i).label.equals("O") ? 0 : 1;
            int predict = h > 0.5 ? 1 : 0;
            
            if (predict == 1 && y == 1) correct ++;
            if (predict == 1) pred ++;
            if (y == 1) gold ++;
        }
        
        double p = (double)correct / pred;
        double r = (double)correct / gold;
        double f = 2.* p * r / (p + r);
        System.out.println("Testing done.");
        System.out.println("");
        System.out.println("# Precision : " + p);
        System.out.println("# Recall: " + r);
        System.out.println("# F: " + f);
    }

}
