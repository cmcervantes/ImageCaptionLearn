package statistical;

import structures.Chain;
import structures.Mention;
import utilities.Logger;
import utilities.StringUtil;
import utilities.Util;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class BCubed extends Score
{
    /**Created a B^3 Score object, where the B^3 score is given as below:
     *
     * Given: M, the set of all mentions
     *        C_g(i), the gold chain containing mention M_i
     *        C_p(i), the predicted chain containing mention M_i
     *        C(i), the set of mentions appearing in both C_g(i) and C_p(i)
     *        w_i, an |M| sized weight vector
     *
     * Precision: \sum_{i}^{|M|} w_i * \frac{|C(i)|}{|C_p(i)|}
     * Precision: \sum_{i}^{|M|} w_i * \frac{|C(i)|}{|C_g(i)|}
     *
     * @param chainSet_pred
     * @param chainSet_gold
     */
    public BCubed(Set<Chain> chainSet_gold,
                  Set<Chain> chainSet_pred)
    {
        //apparently we cannot trust that the mentions
        //inside these chain sets are equivalent objects, so
        //map them to their unique IDs
        Map<String, String> dataClassDict_gold = new HashMap<>();
        for(Chain c : chainSet_gold)
            for(Mention m : c.getMentionSet())
                dataClassDict_gold.put(m.getUniqueID(), c.getUniqueID());
        Map<String, String> dataClassDict_pred = new HashMap<>();
        for(Chain c : chainSet_pred)
            for(Mention m : c.getMentionSet())
                dataClassDict_pred.put(m.getUniqueID(), c.getUniqueID());
        init(dataClassDict_gold, dataClassDict_pred);
    }


    public BCubed(Map<String, String> dataClassDict_gold,
                  Map<String, String> dataClassDict_pred)
    {
        init(dataClassDict_gold, dataClassDict_pred);
    }

    private void init(Map<String, String> dataClassDict_gold,
                      Map<String, String> dataClassDict_pred)
    {
        //add all of the data elements to a set, in case the sets are inequal
        Set<String> dataSet = new HashSet<>(dataClassDict_gold.keySet());

        //create inverted hashmaps, thereby changing
        //      [datum -> class]
        // to
        //      [class -> [data] ]
        Map<String, Set<String>> classDataDict_gold =
                Util.invertMap(dataClassDict_gold);
        Map<String, Set<String>> classDataDict_pred =
                Util.invertMap(dataClassDict_pred);

        _precision = 0.0;
        _recall = 0.0;
        double w = 1.0 / dataSet.size();
        for(String datum : dataSet){
            //get the gold and predicted class sets
            Set<String> data_gold =
                    classDataDict_gold.get(dataClassDict_gold.get(datum));
            Set<String> data_pred =
                    classDataDict_pred.get(dataClassDict_pred.get(datum));

            //if the class wasn't in both sets, continue
            if(data_gold == null || data_pred == null){
                Logger.log("Couldn't find %s in both sets", datum);
                System.out.println(StringUtil.listToString(dataClassDict_gold.keySet(), " | "));
                System.out.println(StringUtil.listToString(dataClassDict_pred.keySet(), " | "));
                System.exit(0);
            }

            //get the intersection of the data sets (how many
            //elements are in this class in both)
            Set<String> dataIntersect = new HashSet<>(data_gold);
            dataIntersect.retainAll(data_pred);

            //w is our normalizer, which is just 1/numMentions
            _precision += w * ((double)dataIntersect.size() /
                    (double)data_pred.size());
            _recall += w * ((double)dataIntersect.size() /
                    (double)data_gold.size());
        }
        setF1();
        type = ScoreType.BCUBED;
    }

    /**Returns a score string in the format
     *  \multicolumn{2}{l}{$B^3$} & PPP.P\% & RRR.R\% & FFF.F\%\\
     *
     * @return
     */
    @Override
    public String toLatexString()
    {
        return "\\multicolumn{2}{l}{$B^3$} & " + super.toLatexString();
    }
}
