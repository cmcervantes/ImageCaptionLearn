package statistical;


import utilities.DoubleDict;
import java.util.HashSet;
import java.util.Set;

public class ScoreDict<K>
{
    private DoubleDict<K> _goldDict;
    private DoubleDict<K> _predDict;
    private DoubleDict<K> _correctDict;
    private Set<K> _keySet;

    public ScoreDict(){
        _goldDict = new DoubleDict<>();
        _predDict = new DoubleDict<>();
        _correctDict = new DoubleDict<>();
        _keySet = new HashSet<>();
    }

    public void increment(K goldLabel, K predLabel)
    {
        _goldDict.increment(goldLabel);
        _predDict.increment(predLabel);
        if(goldLabel.equals(predLabel))
            _correctDict.increment(goldLabel);
        _keySet.add(goldLabel);
        _keySet.add(predLabel);
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
}
