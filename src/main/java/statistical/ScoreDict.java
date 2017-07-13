package statistical;


import utilities.DoubleDict;
import utilities.Logger;
import utilities.StatisticalUtil;
import utilities.StringUtil;

import java.util.*;

public class ScoreDict<K>
{
    private DoubleDict<K> _goldDict;
    private DoubleDict<K> _predDict;
    private DoubleDict<K> _correctDict;
    private Map<K, DoubleDict<K>> _confusion;
    private Set<K> _keySet;
    private List<Double> _goldList;
    private List<Double> _predList;

    public ScoreDict()
    {
        _goldDict = new DoubleDict<>();
        _predDict = new DoubleDict<>();
        _correctDict = new DoubleDict<>();
        _keySet = new HashSet<>();
        _confusion = new HashMap<>();
        _goldList = new ArrayList<>();
        _predList = new ArrayList<>();
    }

    public void increment(ScoreDict<K> scores)
    {
        for(K label : scores._goldDict.keySet())
            _goldDict.increment(label, scores._goldDict.get(label));
        for(K label : scores._predDict.keySet())
            _predDict.increment(label, scores._predDict.get(label));
        for(K label : scores._correctDict.keySet())
            _correctDict.increment(label, scores._correctDict.get(label));
        for(K gold : scores._confusion.keySet()){
            if(!_confusion.containsKey(gold))
                _confusion.put(gold, new DoubleDict<>());
            for(K pred : scores._confusion.get(gold).keySet())
                _confusion.get(gold).increment(pred, scores._confusion.get(gold).get(pred));
        }
        _goldList.addAll(scores._goldList);
        _predList.addAll(scores._predList);
        _keySet.addAll(scores._keySet);
    }

    public void increment(K goldLabel, K predLabel)
    {
        _goldDict.increment(goldLabel);
        _predDict.increment(predLabel);
        if(goldLabel.equals(predLabel))
            _correctDict.increment(goldLabel);
        _keySet.add(goldLabel);
        _keySet.add(predLabel);
        if(!_confusion.containsKey(goldLabel))
            _confusion.put(goldLabel, new DoubleDict<>());
        _confusion.get(goldLabel).increment(predLabel);

        //if this is a numeric label, store the list of
        //predictions / gold labels, so we can do RMSE
        //on request
        Double gold = null;
        Double pred = null;
        if(goldLabel instanceof Double){
            gold = (Double)goldLabel; pred = (Double)predLabel;
        } else if(goldLabel instanceof Integer){
            gold = (double)((Integer)goldLabel); pred = (double)((Integer)predLabel);
        }
        if(gold != null && pred != null){
            _goldList.add(gold); _predList.add(pred);
        }
    }

    public Set<K> keySet(){return _keySet;}

    public Score getScore(K key)
    {
        return new Score((int)_predDict.get(key), (int)_goldDict.get(key),
                (int)_correctDict.get(key));
    }

    public int getGoldCount(K label){ return (int)_goldDict.get(label);}

    public int getTotalGold(){return (int)_goldDict.getSum();}

    /**Returns the accuracy of the classifier; recall that in our world
     * accuracy is measured as the rand index, which can for simplicity
     * be stated as 'the number of samples we got correct over
     * the total number of samples'; Recall further that
     * since we increment the gold and pred dicts simultaneously,
     * summing either will give the total number of examples
     *
     * NOTE: accuracy by label makes no sense, the implicit
     *       assumption in using the rand index is that there are
     *       two classes being compared; using only one class
     *       gets us recall or precision, depending on what we
     *       divide by
     *
     * @return
     */
    public double getAccuracy()
    {
        return 100.0 * _correctDict.getSum() / _goldDict.getSum();
    }

    public double getRMSE() {return StatisticalUtil.computeRMSE(_predList, _goldList);}

    public void printConfusionMatrix()
    {
        List<List<String>> matrix = new ArrayList<>();

        List<K> keys = new ArrayList<>(_keySet);

        //Get the column totals, so we can do percentages
        DoubleDict<K> columnTotals = new DoubleDict<>();
        for(int i=0; i<keys.size(); i++){
            K pred_label = keys.get(i);
            for(int j=0; j<keys.size(); j++){
                K gold_label = keys.get(j);
                int count = 0;
                if(_confusion.containsKey(gold_label))
                    if(_confusion.get(gold_label).containsKey(pred_label))
                        count = (int)_confusion.get(gold_label).get(pred_label);
                columnTotals.increment(gold_label, count);
            }
        }
        //We want to ignore all columns without any gold labels
        Set<Integer> ignoredColumnIndices = new HashSet<>();
        for(int i=0; i<keys.size(); i++)
            if(columnTotals.get(keys.get(i)) == 0)
                ignoredColumnIndices.add(i);

        //Add column headers (gold labels)
        List<String> columnHeaders = new ArrayList<>();
        for(int i=0; i<keys.size(); i++)
            if(!ignoredColumnIndices.contains(i))
                columnHeaders.add(keys.get(i).toString());
        columnHeaders.add(0, ""); //add an empty cell in the top left
        matrix.add(columnHeaders);

        //Iterate through the keys in a pred/gold fashion to get each row
        for(int i=0; i<keys.size(); i++){
            List<String> row = new ArrayList<>();

            //Add the pred label to the start of the row
            K pred_label = keys.get(i);
            row.add(pred_label.toString());

            for(int j=0; j<keys.size(); j++){
                if(ignoredColumnIndices.contains(j))
                    continue;

                K gold_label = keys.get(j);
                int count = 0;
                if(_confusion.containsKey(gold_label))
                    if(_confusion.get(gold_label).containsKey(pred_label))
                        count = (int)_confusion.get(gold_label).get(pred_label);
                row.add(String.format("%d (%2.1f%%)", count,
                        100.0 * count / columnTotals.get(gold_label)));
            }
            matrix.add(row);
        }

        System.out.println(StringUtil.toTableStr(matrix, true));
        System.out.println("Table total: " + columnTotals.getSum());
    }

    /**Prints the complete score output (reused frequently enough as to require
     * its own function), including confusion matrix, latex table, and accuracy
     *
     */
    public void printCompleteScores()
    {
        Logger.log("Confusion Matrix");
        printConfusionMatrix();

        Logger.log("Scores");
        for(K label : _keySet){
            int goldCount = getGoldCount(label);
            if(goldCount > 0)
                System.out.printf("%s & %.2f\\%% & %s \\\\\n", label.toString(),
                        100.0 * (double)goldCount / getTotalGold(),
                        getScore(label).toLatexString());
        }
        Logger.log("Accuracy: %.2f%%", getAccuracy());
    }
}
