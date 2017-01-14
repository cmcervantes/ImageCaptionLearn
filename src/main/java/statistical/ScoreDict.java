package statistical;


import utilities.DoubleDict;

import java.util.*;

public class ScoreDict<K>
{
    private DoubleDict<K> _goldDict;
    private DoubleDict<K> _predDict;
    private DoubleDict<K> _correctDict;
    private Map<K, DoubleDict<K>> _confusion;
    private Set<K> _keySet;

    public ScoreDict(){
        _goldDict = new DoubleDict<>();
        _predDict = new DoubleDict<>();
        _correctDict = new DoubleDict<>();
        _keySet = new HashSet<>();
        _confusion = new HashMap<>();
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
    }

    public Set<K> keySet(){return _keySet;}

    public Score getScore(K key)
    {
        return new Score((int)_predDict.get(key), (int)_goldDict.get(key),
                (int)_correctDict.get(key));
    }

    public int getGoldCount(K label){ return (int)_goldDict.get(label);}

    public int getTotalGold(){return (int)_goldDict.getSum();}

    public double getAccuracy()
    {
        return 100.0 * _correctDict.getSum() / _goldDict.getSum();
    }

    public double getAccuracy(K label)
    {
        return 100.0 * _correctDict.get(label) / _goldDict.get(label);
    }

    public void printConfusionMatrix()
    {
        //get the largest count, for formatting
        int maxLen = 0;
        for(K gold : _confusion.keySet()) {
            for (K pred : _confusion.get(gold).keySet()) {
                int len = String.valueOf(_confusion.get(gold).get(pred)).length();
                if(len > maxLen)
                    maxLen = len;
            }
        }
        maxLen += 8;

        //store the keys as a list for consistent ordering
        List<K> keys = new ArrayList<>(_keySet);

        //column headers are formatted as
        //      | gold  gold  ... gold
        //and each row will be formatted as
        // pred | count count ... count
        String formatStr = "%-2s | ";
        for(int i=0; i<keys.size(); i++)
            formatStr += "%-" + String.valueOf(maxLen) + "s ";
        formatStr += "\n";

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

        List<String> colHeaders = new ArrayList<>();
        colHeaders.add(" ");
        List<List<String>> matrix = new ArrayList<>();
        for(int i=0; i<keys.size(); i++){
            K pred_label = keys.get(i);

            //since both pred and gold labels are in order,
            //we're adding column headers as we go through the rows
            colHeaders.add(pred_label.toString());

            //we also want to add this label to the start of the row
            List<String> row = new ArrayList<>();
            row.add(pred_label.toString());
            for(int j=0; j<keys.size(); j++){
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
        matrix.add(0, colHeaders);

        for(List<String> row : matrix)
            System.out.printf(formatStr, row.toArray());
    }
}
