package statistical;

import structures.Chain;
import structures.Mention;
import utilities.StringUtil;

import java.util.*;

/**The Blanc class allows us to compute BLANC
 * scores, which contains _f1 for both
 * positive and negative links, as well as the
 * mean of both
 *
 * Special cases from the original paper
 * (adjusted for our terminology)
 * 1) If PRED has only one chain...
 *      a) If GOLD contains only one chain: 100
 *      b) If GOLD is only singletones: 0
 *      c) If GOLD is mixed: negative scores are 0
 * 2) If PRED contains only singletones...
 *      a) If GOLD is only singletones: 100
 *      b) If GOLD is only one chain: 0
 *      c) If GOLD is mixed: positive scores are 0
 * 3) If GOLD contains links of both types but
 *    there are no correct positive links,
 *    positive scores are 0; if no correct negative
 *    links, negative scores are 0
 *
 * Original paper can be found [here][orig_link]
 * [orig_link]: https://www.researchgate.net/profile/Eduard_Hovy2/publication/231881781_BLANC_Implementing_the_Rand_index_for_coreference_evaluation/links/553122420cf2f2a588acdc95.pdf
 *
 * We also make the augmentation for mismatched key and
 * response mentions; as follows
 *
 *                     |Ck ∩ Cr|
 *      Rc = ----------------------------------
 *             |Ck ∩ Cr| + |Ck ∩ Nr| + |Ck \ Tr|
 *
 *                     |Ck ∩ Cr|
 *      Pc = ------------------------------------
 *             |Cr ∩ Ck| + |Cr ∩ Nk| + |Cr \ Tk|
 *
 *              2 Rc Pc
 *      Fc = ---------------
 *              Rc + Pc
 *
 *                      |Nk ∩ Nr|
 *      Rn = ---------------------------------
 *            |Nk ∩ Cr| + |Nk ∩ Nr| + |Nk \ Tr|
 *
 *                  |Nk ∩ Nr|
 *      Pn =  -------------------------------------
 *             |Nr ∩ Ck| + |Nr ∩ Nk| + |Nr \ Tk|
 *
 *              2 Rn Pn
 *      Fn = --------------
 *              Rn + Pn
 *
 *  Where
 *      Ck: The set of coreference links in the key (gold)
 *      Cr: The set of coreference links in the response (pred)
 *      Nk: The set of noncoreference links in the key
 *      Nr: The set of noncoreference links in the response
 *      Ck \ Tr: The set of coreference links present in the key
 *               and missing in the response
 *      Nk \ Tr: The set of noncoreference links present in the key
 *               and missing in the response
 *      Cr \ Tk: The set of coreference links present in the response
 *               and missing in the key
 *      Nr \ Tk: The set of noncoreference links prsent in the response
 *               and missing in the key
 *
 * Augmentation paper can be found [here][aug_paper]
 * [aug_paper] : http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42559.pdf
 *
 * @author ccervantes
 */
public class Blanc extends Score
{
	private Score score_neg;
	private Score score_pos;
	
	/**Constructor requires dictionaries for
	 * predicted chain assignments (<b>chainDict_pred</b>) 
	 * and gold chain assignments (<b>chainDict_gold</b>)
	 */
	public Blanc(Set<Chain> chainSet_gold,
                 Set<Chain> chainSet_pred)
	{
        Map<String, String> mentionChainDict_pred = new HashMap<>();
        Map<String, String> mentionChainDict_gold = new HashMap<>();
        for(Chain c : chainSet_pred)
            for(Mention m : c.getMentionSet())
                mentionChainDict_pred.put(m.getUniqueID(), c.getUniqueID());
        for(Chain c : chainSet_gold)
            for(Mention m : c.getMentionSet())
                mentionChainDict_gold.put(m.getUniqueID(), c.getUniqueID());
        init(mentionChainDict_gold, mentionChainDict_pred);
	}

    public Blanc(Map<String, String> dataClassDict_gold,
                 Map<String, String> dataClassDict_pred)
    {
        init(dataClassDict_gold, dataClassDict_pred);
    }

	private Blanc(Score negScore, Score posScore)
	{
		super(negScore, posScore);
		score_neg = negScore;
		score_pos = posScore;
        type = ScoreType.BLANC;
	}

    private void init(Map<String, String> dataClassDict_gold,
                      Map<String, String> dataClassDict_pred)
    {
        //Store the full graph according to both predicted and gold
        //partitions (true for a positive link; false if a negative link)
        Map<String, Boolean> dataPairLinkDict_gold = getDataLinkDict(dataClassDict_gold);
        Map<String, Boolean> dataPairLinkDict_pred = getDataLinkDict(dataClassDict_pred);

        //Create our mention sets, based on this full graph
        Set<String> ck = new HashSet<>();
        Set<String> nk = new HashSet<>();
        for(String dataPair : dataPairLinkDict_gold.keySet()){
            if(dataPairLinkDict_gold.get(dataPair))
                ck.add(dataPair);
            else
                nk.add(dataPair);
        }
        Set<String> cr = new HashSet<>();
        Set<String> nr = new HashSet<>();
        for(String dataPair : dataPairLinkDict_pred.keySet()){
            if(dataPairLinkDict_pred.get(dataPair))
                cr.add(dataPair);
            else
                nr.add(dataPair);
        }

        //Compute the relevant intersections and div sets
        Set<String> ck_cr = new HashSet<>(ck);
        ck_cr.retainAll(cr);    // Ck ∩ Cr
        Set<String> ck_nr = new HashSet<>(ck);
        ck_nr.retainAll(nr);    // Ck ∩ Nr
        Set<String> ck_tr = new HashSet<>(ck);
        ck_tr.removeAll(cr);    // Ck \ Tr
        Set<String> cr_nk = new HashSet<>(cr);
        cr_nk.retainAll(nk);    // Cr ∩ Nk
        Set<String> cr_tk = new HashSet<>(cr);
        cr_tk.removeAll(ck);    // Cr \ Tk
        Set<String> nk_nr = new HashSet<>(nk);
        nk_nr.retainAll(nr);    // Nk ∩ Nr
        Set<String> nk_cr = new HashSet<>(nk);
        nk_cr.retainAll(cr);    // Nk ∩ Cr
        Set<String> nk_tr = new HashSet<>(nk);
        nk_tr.removeAll(nr);    // Nk \ Tr
        Set<String> nr_ck = new HashSet<>(nr);
        nr_ck.retainAll(ck);    // Nr ∩ Ck
        Set<String> nr_tk = new HashSet<>(nr);
        nr.removeAll(nk);       // Nr \ Tk

        //Finally, compute positive and negative coref scores
        double rc = (double)ck_cr.size() /
                    (double)(ck_cr.size() + ck_nr.size() + ck_tr.size());
        double pc = (double)ck_cr.size() /
                    (double)(ck_cr.size() + cr_nk.size() + cr_tk.size());
        score_pos = new Score(pc, rc);
        double rn = (double)nk_nr.size() /
                    (double)(nk_cr.size() + nk_nr.size() + nk_tr.size());
        double pn = (double)nk_nr.size() /
                    (double)(nr_ck.size() + nk_nr.size() + nr_tk.size());
        score_neg = new Score(pn, rn);

        init(score_pos, score_neg);
        type = ScoreType.BLANC;
    }
	
	/**Getters*/
	public double getRecall_pos(){return score_pos._recall;}
	public double getPrecision_pos(){return score_pos._precision;}
	public double getAccuracy_pos(){return score_pos._f1;}
	public double getRecall_neg(){return score_neg._recall;}
	public double getPrecision_neg(){return score_neg._precision;}
	public double getAccuracy_neg(){return score_neg._f1;}

	/**Returns the overall score string (average 
	 * of positive and negative scores)
	 * 
	 * @return - The overall score string
	 */
 	public String getScoreString_total()
	{
		return toScoreString();
	}
	
 	/**Returns the score string for the positive 
 	 * links
 	 * 
 	 * @return - The positive score string
 	 */
	public String getScoreString_pos()
	{
		return score_pos.toScoreString();
	}
	
	/**Returns the score string for the negative
	 * links
	 * 
	 * @return - The negative score string
	 */
	public String getScoreString_neg()
	{
		return score_neg.toScoreString();
	}


    /**Returns a score string in the format
     *      \multicolumn{2}{l}{BLANC} & PPP.P\% & RRR.R\% & FFF.F\%\\
     *      & Pos & PPP.P\% & RRR.R\% & FFF.F\%\\
     *      & Neg & PPP.P\% & RRR.R\% & FFF.F\%\\
     *
     * @return
     */
    @Override
    public String toLatexString()
    {
        String s = "\\multicolumn{2}{l}{BLANC} & " + super.toLatexString() + "\n";
        s += "& Pos & " + score_pos.toLatexString() + "\n";
        s += "& Neg & " + score_neg.toLatexString();
        return s;
    }

	/**Returns a Blanc score representing the average score of those
	 * found in <b>scores</b>, where each category is
	 * averaged separately
	 * 
	 * @param scores - The scores to average
	 * @return		 - The single average score
	 */
	public static Blanc getAverageBlanc(Collection<Blanc> scores)
	{
		Score[] scoreArr_neg = new Score[scores.size()];
		Score[] scoreArr_pos = new Score[scores.size()];
		int i = 0;
		for(Blanc b : scores) {
			scoreArr_neg[i] = b.score_neg;
			scoreArr_pos[i] = b.score_pos;
			i++;
		}
		Score negScore = new Score(scoreArr_neg);
		Score posScore = new Score(scoreArr_pos);
		return new Blanc(negScore, posScore);
	}

    private static Map<String, Boolean> getDataLinkDict(Map<String, String> dataClassDict)
    {
        Map<String, Boolean> dataPairLinkDict = new HashMap<>();
        List<String> dataList = new ArrayList<>(dataClassDict.keySet());
        for(int i=0; i<dataList.size(); i++){
            for(int j=i+1; j<dataList.size(); j++){
                String datum_1 = dataList.get(i);
                String datum_2 = dataList.get(j);
                String dataPair = StringUtil.getAlphabetizedPair(datum_1, datum_2);
                boolean eq = false;
                if(dataClassDict.containsKey(datum_1) && dataClassDict.containsKey(datum_2))
                    eq = dataClassDict.get(datum_1).equals(dataClassDict.get(datum_2));
                dataPairLinkDict.put(dataPair, eq);
            }
        }
        return dataPairLinkDict;
    }
}
