package learn;

import libsvm.*;
import statistical.Score;
import utilities.DoubleDict;
import utilities.FileIO;
import utilities.Logger;

import java.io.IOException;
import java.util.*;

public class SVM
{
    private svm_parameter params;
    private int crossValFolds;
    private svm_model model;

    public SVM(SVM_TYPE svmType, KERNEL_TYPE kernelType, int kernelDegree,
               int kernelCoef, int cost, double nu, double epsilon_loss,
               double epsilon_thresh, int[] weightLabels, double[] weights,
               boolean useShrinking, boolean estimateProbs, int nFolds,
               int cache)
    {
        init(svmType, kernelType, kernelDegree, kernelCoef, cost,
             nu, epsilon_loss, epsilon_thresh, weightLabels, weights,
             useShrinking, estimateProbs, nFolds, cache);
    }

    public SVM(SVM_TYPE svm_type, KERNEL_TYPE kernelType, int cache)
    {
        //set with the defaults present in the libsvm examples
        init(svm_type, kernelType, 3, 0, 1, 1000,
             0.1, 0.1, new int[0], new double[0], false,
             true, -1, cache);
    }

    /**Initialized the SVM, according to
     *
     * -d degree : set degree in kernel function (default 3)
     * -g gamma : set gamma in kernel function (default 1/num_features)
     * -r coef0 : set coef0 in kernel function (default 0)
     * -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
     * -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
     * -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
     * -m cachesize : set cache memory size in MB (default 100)
     * -e epsilon : set tolerance of termination criterion (default 0.001)
     * -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
     * -b probability_estimates : whether to train_excludeIndices a SVC or SVR model for probability estimates, 0 or 1 (default 0)
     * -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
     * -v n : n-fold cross validation mode
     * -q : quiet mode (no outputs)
     *
     * @param svmType
     * @param kernelType
     * @param kernelDegree
     * @param kernelCoef
     * @param cost
     * @param nu
     * @param epsilon_loss
     * @param epsilon_thresh
     * @param weightLabels
     * @param weights
     * @param useShrinking
     * @param estimateProbs
     * @param nFolds
     * @param cache
     */
    private void init(SVM_TYPE svmType, KERNEL_TYPE kernelType,
          int kernelDegree, int kernelCoef, int cost, double nu,
          double epsilon_loss, double epsilon_thresh, int[] weightLabels,
          double[] weights, boolean useShrinking, boolean estimateProbs,
          int nFolds, int cache)
    {
        //set the parameters in the svm params
        params = new svm_parameter();
        params.svm_type = svmType.getSvmParamCode();
        params.kernel_type = kernelType.getSvmParamCode();
        params.degree = kernelDegree;
        params.coef0 = kernelCoef;
        params.nu = nu;
        params.cache_size = cache;
        params.C = cost;
        params.eps = epsilon_thresh;
        params.p = epsilon_loss;
        params.shrinking = useShrinking ? 1 : 0;
        params.probability = estimateProbs ? 1 : 0;
        params.weight_label = weightLabels;
        params.weight = weights;
        //params.nr_weight++;


        //set our local vars
        crossValFolds = nFolds;


        svm.svm_set_print_string_function(svm_print_null);
    }

    public void train(String filename)
    {
        //get the feature vector set (we assume svmlite formatting)
        Set<FeatureVector> fvSet = new HashSet<>();
        List<String> lineList = FileIO.readFile_lineList(filename);
        for(String line : lineList) {
            FeatureVector fv = FeatureVector.parseFeatureVector(line);
            fvSet.add(fv);
        }


        //convert these feature vectors to our needed svm_node structures
        Vector<Double> labelVector = new Vector<>();
        Vector<svm_node[]> nodeArrVector = new Vector<>();
        int maxIdx = 0;
        for(FeatureVector fv : fvSet){
            List<Integer> featureIndices = fv.getFeatureIndices();
            labelVector.add(fv.label);
            svm_node[] nodeArr = new svm_node[featureIndices.size()];
            int i=0;
            for(Integer featIdx : featureIndices){
                nodeArr[i] = new svm_node();
                nodeArr[i].index = featIdx;
                nodeArr[i].value = fv.getFeatureValue(featIdx);

                if(featIdx > maxIdx)
                    maxIdx = featIdx;
                i++;
            }
            nodeArrVector.add(nodeArr);
        }

        //set up an svm problem
        svm_problem problem = new svm_problem();
        problem.l = labelVector.size();
        problem.x = new svm_node[problem.l][];
        for(int i=0; i<problem.l; i++)
            problem.x[i] = nodeArrVector.get(i);
        problem.y = new double[problem.l];
        for(int i=0; i<problem.l; i++)
            problem.y[i] = labelVector.get(i);
        params.gamma = 1.0 / maxIdx;

        //train_excludeIndices the model, either simply or with crossval
        if(crossValFolds > 0) {
            int total_correct = 0;
            double total_error = 0;
            double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
            double[] target = new double[problem.l];

            svm.svm_cross_validation(problem,params,crossValFolds,target);
            if(params.svm_type == svm_parameter.EPSILON_SVR ||
               params.svm_type == svm_parameter.NU_SVR) {
                for(int i=0;i<problem.l;i++)
                {
                    double y = problem.y[i];
                    double v = target[i];
                    total_error += (v-y)*(v-y);
                    sumv += v;
                    sumy += y;
                    sumvv += v*v;
                    sumyy += y*y;
                    sumvy += v*y;
                }
                System.out.print("Cross Validation Mean squared error = "+total_error/problem.l+"\n");
                System.out.print("Cross Validation Squared correlation coefficient = "+
                        ((problem.l*sumvy-sumv*sumy)*(problem.l*sumvy-sumv*sumy))/
                                ((problem.l*sumvv-sumv*sumv)*(problem.l*sumyy-sumy*sumy))+"\n"
                );
            } else {
                for(int i=0;i<problem.l;i++)
                    if(target[i] == problem.y[i])
                        ++total_correct;
                System.out.print("Cross Validation Accuracy = "+100.0*total_correct/problem.l+"%\n");
            }
        } else {
            model = svm.svm_train(problem, params);
        }
    }

    public void saveModel(String filename)
    {
        try{
            svm.svm_save_model(filename,model);
        } catch (IOException ioEx) {
            Logger.log(ioEx);
        }
    }

    public void loadModel(String filename)
    {
        try{
            model = svm.svm_load_model(filename);
            params = model.param;
        } catch(IOException ioEx) {
            Logger.log(ioEx);
        }
    }

    public Map<FeatureVector, Double> predict(String filename)
    {
        int correct = 0;
        int total = 0;
        double error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        Map<FeatureVector, Double> featVectorLabelDict =
                new HashMap<>();


        int svm_type=svm.svm_get_svm_type(model);
        int nr_class=svm.svm_get_nr_class(model);
        double[] prob_estimates=null;

        if(params.probability == 1) {
            //NOTE: if type != epsilon_svr or mu_svr

            //retrieve the labels from the model
            int[] labels = new int[nr_class];
            svm.svm_get_labels(model, labels);

            //print them?
            prob_estimates = new double[nr_class];
            System.out.print("labels: ");
            for(int i=0; i<nr_class; i++)
                System.out.print(labels[i] + " ");
            System.out.println();
        }

        //read the prediction feature file
        Set<FeatureVector> fvSet = new HashSet<>();
        List<String> lineList = FileIO.readFile_lineList(filename);
        for(String line : lineList) {
            FeatureVector fv = FeatureVector.parseFeatureVector(line);
            fvSet.add(fv);
        }

        //predict a label for each feature vector
        DoubleDict<String> countDict = new DoubleDict<>();
        for(FeatureVector fv : fvSet){
            List<Integer> featureIndices = fv.getFeatureIndices();

            //read the feature vector as our x
            svm_node[] x = new svm_node[featureIndices.size()];
            int i=0;
            for(Integer featIdx : featureIndices){
                x[i] = new svm_node();
                x[i].index = featIdx;
                x[i].value = fv.getFeatureValue(featIdx);
                i++;
            }

            //get the best label, yhat
            double yhat;
            if(params.probability == 1){
                yhat = svm.svm_predict_probability(model, x, prob_estimates);
                System.out.print(yhat + " ");
                for(int j=0; j<nr_class; j++)
                    System.out.print(j + ": " + prob_estimates[j] + " ");
                System.out.println();
            } else {
                yhat = svm.svm_predict(model, x);
            }
            featVectorLabelDict.put(fv, yhat);

            //compare our predicted label to our true label, y
            double y = fv.label;

            if(y==1)
                countDict.increment("gold_pos");
            else
                countDict.increment("gold_neg");

            if(yhat == 1)
                countDict.increment("pred_pos");
            else
                countDict.increment("pred_neg");

            if(yhat == y){
                if(y == 0)
                    countDict.increment("correct_neg");
                else
                    countDict.increment("correct_pos");
                correct++;
            }
            error += Math.pow(yhat-y, 2);
            sumv += yhat;
            sumy += y;
            sumyy += Math.pow(y, 2);
            sumvy += yhat * y;
            total++;
        }

        Score s_pos = new Score((int)countDict.get("pred_pos"),
                (int)countDict.get("gold_pos"), (int)countDict.get("correct_pos"));
        Score s_neg = new Score((int)countDict.get("pred_neg"),
                (int)countDict.get("gold_neg"), (int)countDict.get("correct_neg"));

        Logger.log("Pos: " + s_pos.toScoreString());
        Logger.log("Neg: " + s_neg.toScoreString());

        /*
        Logger.log("Mean squared error = "+error/total+" (regression)");
        Logger.log("Squared correlation coefficient = "+
                ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
                        ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))+
                " (regression)");
        Logger.log("Accuracy = "+(double)correct/total*100+
                "% ("+correct+"/"+total+") (classification)");*/
        /*

        svm_predict.info("Mean squared error = "+error/total+" (regression)\n");
        svm_predict.info("Squared correlation coefficient = "+
                ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
                        ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))+
                " (regression)\n");

        svm_predict.info("Accuracy = "+(double)correct/total*100+
                "% ("+correct+"/"+total+") (classification)\n");

        */

        return featVectorLabelDict;
    }


    private static svm_print_interface svm_print_null = new svm_print_interface()
    {
        public void print(String s) {Logger.log(s);}
    };

    /**Denotes the type of SVM to use, according to
     *
     * 0 -- C-SVC		(multi-class classification)
     * 1 -- nu-SVC		(multi-class classification)
     * 2 -- one-class SVM
     * 3 -- epsilon-SVR	(regression)
     * 4 -- nu-SVR		(regression)
     */
    public enum SVM_TYPE
    {
        C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR;

        public int getSvmParamCode()
        {
            switch(this)
            {
                case C_SVC: return svm_parameter.C_SVC;
                case NU_SVC: return svm_parameter.NU_SVC;
                case ONE_CLASS: return svm_parameter.ONE_CLASS;
                case EPSILON_SVR: return svm_parameter.EPSILON_SVR;
                case NU_SVR: return svm_parameter.NU_SVR;
            }
            return -1;
        }

        @Override
        public String toString()
        {
            switch(this)
            {
                case C_SVC: return "C_SVC";
                case NU_SVC: return "NU_SVC";
                case ONE_CLASS: return "ONE_CLASS";
                case EPSILON_SVR: return "EPSILON_SVR";
                case NU_SVR: return "NU_SVR";
            }
            return "";
        }

        public static SVM_TYPE parseType(String s)
        {
            s = s.toUpperCase();
            switch(s)
            {
                case "C_SVC": return C_SVC;
                case "NU_SVC": return NU_SVC;
                case "ONE_CLASS": return ONE_CLASS;
                case "EPSILON_SVR": return EPSILON_SVR;
                case "NU_SVR": return NU_SVR;
            }
            return null;
        }
    }

    /**Denotes the type of kernel to use, according to
     *
     *  0 -- linear: u'*v
     *	1 -- polynomial: (gamma*u'*v + coef0)^degree
     *	2 -- radial basis function: exp(-gamma*|u-v|^2)
     *	3 -- sigmoid: tanh(gamma*u'*v + coef0)
     *	4 -- precomputed kernel (kernel values in training_set_file)
     */
    public enum KERNEL_TYPE
    {
        LINEAR, POLY, RBF, SIGMOID, PRECOMP;

        public int getSvmParamCode()
        {
            switch(this)
            {
                case LINEAR: return svm_parameter.LINEAR;
                case POLY: return svm_parameter.POLY;
                case RBF: return svm_parameter.RBF;
                case SIGMOID: return svm_parameter.SIGMOID;
                case PRECOMP: return svm_parameter.PRECOMPUTED;
            }
            return -1;
        }

        @Override
        public String toString()
        {
            switch(this)
            {
                case LINEAR: return "LINEAR";
                case POLY: return "POLY";
                case RBF: return "RBF";
                case SIGMOID: return "SIGMOID";
                case PRECOMP: return "PRECOMP";
            }
            return "";
        }

        public static KERNEL_TYPE parseType(String s)
        {
            s = s.toUpperCase();
            switch(s)
            {
                case "LINEAR": return LINEAR;
                case "POLY": return POLY;
                case "RBF": return RBF;
                case "SIGMOID": return SIGMOID;
                case "PRECOMP": return PRECOMP;
            }
            return null;
        }

    }
}
