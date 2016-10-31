package learn;

import structures.*;
import utilities.DoubleDict;

import java.util.*;

/**CorefChainThreads performs all our predicted
 * coreference chain logic for an image
 * 
 * @author ccervantes
 */
public class ChainThread extends Thread
{
	/* NOTE: Creating predicted coreference chains is an
	 * O(n^4) operation. Lie. It's an O(m*n*o*p) 
	 * operation. And the more I thought about it
	 * the more it occurred to me that while I could
	 * change the operation to use fewer loops, all I'd
	 * be doing was making it an O(mn*op) operation or
	 * O(m*no*p) operation. Without changing the inference
	 * logic at its main.java.core or the structure of our data set,
	 * there was nothing much I could do about this.
	 * Hence, the CorefChainThread class.
	 * Chains for an image can be created entirely
	 * independently from other images. Thus, our
	 * O(n^4) operation is now parallelized along 
	 * the largest of those dimensions, which in 
	 * practice makes this very manageable. 
	 */
	private double _confThresh;
	private List<String> _debugOutList;
	private Collection<Mention> _mentionSet;
    private Set<Chain> _predChainSet;
    private ChainClassifier.InferenceType _infType;
    private BinaryClassifierScoreDict _scoreDict;
    private String _docID;

	public ChainThread(Collection<Mention> mentionSet, double confThresh,
                       ChainClassifier.InferenceType infType,
					   BinaryClassifierScoreDict scoreDict)
	{
        _mentionSet = mentionSet;
        //get the document ID from a random mention
        for(Mention m : _mentionSet){
            _docID = m.getDocID();
            break;
        }
		_confThresh = confThresh;
        _infType = infType;
        if(scoreDict == null)
            _scoreDict = new BinaryClassifierScoreDict();
        else
            _scoreDict = scoreDict; //keep a reference, don't copy it
		_predChainSet = new HashSet<>();
		_debugOutList = new ArrayList<>();
	}

	public void run()
	{
        switch(_infType){
            case GREEDY: inference_greedy();
                break;
        }
	}

    /**Runs the thread using the following predicted chain
     * inference logic:
     * 1) Add the next mention to the best chain
     * 2) If we can't add it to a chain, make the strongest
     *    unassigned pair their own chain
     * 3) If we can't do that, make this mention a singleton chain
     */
	private void inference_greedy()
    {
        //until we've assigned all the mentions
        while(_mentionSet.size() > 0) {
            //on the last iteration of this loop, we want to know
            //why the remaining mentions aren't in a chain
            ArrayList<String> outList = new ArrayList<>();
            outList.add("Confidence with which mentions can be added to chains");

            //1) compute the confidences of all possible mention-chain additions
            double maxConfidence = Double.NEGATIVE_INFINITY;
            Mention bestMention = null;
            Chain bestChain = null;
            for(Mention mentionToAdd : _mentionSet) {
                //mentions can only be validly added to chains if they adhere to
                //two constraints:
                //a) the mention's caption isn't already in the chain
                //b) the mention's type matches those of the other mentions in the chain
                Integer capToAdd = mentionToAdd.getCaptionIdx();
                String typeToAdd = mentionToAdd.getLexicalType();

                //iterate through each element we've already put into
                //a chain, computing the confidence of adding
                //this new mention to an existing chain
                for(Chain c : _predChainSet) {
                    boolean validAddition = true;
                    double totalConfidence = 0.0;
                    DoubleDict<Mention> mentionScoreDict = new DoubleDict<>();
                    for(Mention mentionInChain : c.getMentionSet()) {
                        //rather than search for which order these mentions
                        //are in, let's just do two constant time lookups
                        //in our mentionPair / score dict, assuming
                        //as we do that AB and BA aren't _both_ there
                        Double score = _scoreDict.get(mentionToAdd, mentionInChain);
                        if(score != null){
                            totalConfidence += score;
                            mentionScoreDict.increment(mentionInChain, score);

                            //if this is one of our special cases where
                            //either the score is max or min val
                            //(the signal being: must have this link or must not)
                            //address that in the valid addition
                            if(score == Double.MAX_VALUE){
                                validAddition = true;
                            } else if(score == Double.MIN_VALUE){
                                validAddition = false;
                            } else {
                                validAddition &= mentionToAdd.getCaptionIdx() != mentionInChain.getCaptionIdx();
                                //ignore pronoun lexical types, since they're basically trash
                                if(mentionInChain.getPronounType() == Mention.PRONOUN_TYPE.NONE)
                                    validAddition &= typeToAdd.equals(mentionInChain.getLexicalType());
                            }

                        } else {
                            validAddition = false;
                        }
                    }

                    //we're using the mean inference mechanism, so normalize the score
                    if(validAddition) {
                        double conf = totalConfidence / c.getMentionSet().size();
                        outList.add(String.format(
                                "Mention %s, chain [%s] with conf: [%.4f]",
                                mentionToAdd.toString(), c.getID(), conf));

                        conf = mentionScoreDict.getMax();

                        if(conf > _confThresh && conf > maxConfidence) {
                            maxConfidence = conf;
                            bestMention = mentionToAdd;
                            bestChain = c;
                        }
                    }
                }
            }

            //if we have a mention and a chain index, assign this mention
            //to that chain!
            if(bestChain != null && bestMention != null) {
                _debugOutList.add(String.format(
                        "Adding %s to [%s] with confidence: %.4f\n",
                        bestMention.toString(), bestChain.getID(),
                        maxConfidence));
                bestChain.addMention(bestMention);

                //also remove this mention from the set, and continue
                _mentionSet.remove(bestMention);
                continue;
            }

            //Reaching here means we didn't find a mention that
            //we could add to a pre-existing chain. No worries!
            //Let's try to make a new one from a link
            outList.add("Remaining mention pair confidences");

            //2) find the menion pair with the highest link probability
            //   from those mentions we haven't yet assigned,
            //   being mindful of our constraints
            maxConfidence = Double.NEGATIVE_INFINITY;
            Mention bestMention_1 = null;
            Mention bestMention_2 = null;
            List<Mention> mentionList = new ArrayList<>(_mentionSet);
            for(int i=0; i<mentionList.size(); i++){
                for(int j=i+1; j<mentionList.size(); j++){
                    Mention m1 = mentionList.get(i);
                    Mention m2 = mentionList.get(j);
                    Double score = _scoreDict.get(m1, m2);
                    if(score != null){
                        boolean validLink;
                        if(score == Double.MAX_VALUE) {
                            validLink = true;
                        } else if(score == Double.MIN_VALUE) {
                            validLink = false;
                        } else {
                            validLink = m1.getLexicalType().equals(m2.getLexicalType()) &&
                                        m1.getCaptionIdx() != m2.getCaptionIdx();
                        }
                        if(validLink){
                            //add this mention pair in case
                            //this is the last iteration
                            outList.add(String.format("[%s | %s] : %.4f",
                                    m1.toString(), m2.toString(), score));

                            if(score > maxConfidence &&
                                    score > _confThresh) {
                                //store thse two mentions. They are the best.
                                //The best!
                                bestMention_1 = m1;
                                bestMention_2 = m2;
                                maxConfidence = score;
                            }
                        }
                    }
                }
            }

            //if we have two mentions, we know they're our highest link
            //from among our remaining mentions AND that they
            //don't belong in any other cluster AND
            //their link confidence exceeds our threshold. They can now
            //be happily placed in their own chain
            if(bestMention_1 != null && bestMention_2 != null) {
                _debugOutList.add(String.format(
                        "Making chain [%d] from pair [%s | %s] with confidence: %.3f\n",
                        _predChainSet.size(), bestMention_1.getUniqueID(),
                        bestMention_2.getUniqueID(), maxConfidence));

                //create a new chain, add these mentions to it, and
                //add it to the set
                Chain c = new Chain(bestMention_1.getDocID(), String.valueOf(_predChainSet.size()));
                c.addMention(bestMention_1);
                c.addMention(bestMention_2);
                _predChainSet.add(c);

                //pop those two mentions off the set
                _mentionSet.remove(bestMention_1);
                _mentionSet.remove(bestMention_2);

                //and move on
                continue;
            }

            //Reaching _here_ means that not only did we not find
            //any mention that can be added with high enough confidence
            //to any pre-existing chain, but we _also_ failed to find
            //any pair from among the to-add list that had a high enough
            //link confidence to start a chain of their own. What does
            //this mean? This means that we're only left with singletons
            //who don't have any likelihood to connect with anyone. It's
            //all very sad. But fearnot! They get to be their own chains!

            //3) add all remaining unassigned mentions as their own chains
            for(Mention m : _mentionSet) {
                Chain c = new Chain(m.getDocID(), String.valueOf(_predChainSet.size()));
                c.addMention(m);
                _predChainSet.add(c);
            }

            //this was our last iteration. so we want to add the output
            //from all those earlier things and add them to our actual
            //output
            _debugOutList.add(String.format("Added %d singleton chains",
                    _mentionSet.size()));
            _debugOutList.addAll(outList);

            _mentionSet = new HashSet<>();
        }
    }

    /**Getters*/
    public String getDocID(){return _docID;}
	public List<String> getDebugOutput(){return _debugOutList;}
    public Set<Chain> getPredictedChains(){return _predChainSet;}
}
