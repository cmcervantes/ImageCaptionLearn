package learn;

import edu.illinois.cs.cogcomp.lbjava.infer.GurobiHook;
import structures.Chain;
import structures.Mention;
import utilities.Logger;
import utilities.Util;

import java.util.*;


/**Implementation of the ILP solver used for multithreaded coreference
 * inference, which wraps the CogComp ILP solver;
 * large sections of code borrowed from [Dan's group][link].
 * [link]: https://github.com/xiaoling/wikifier/blob/master/src/edu/illinois/cs/cogcomp/lbj/coref/decoders/ILPDecoder.java
 */
public class ILPSolverThread extends Thread
{
    private edu.illinois.cs.cogcomp.lbjava.infer.ILPSolver _solver;
    private BinaryClassifierScoreDict _scoreDict_coref;
    private BinaryClassifierScoreDict _scoreDict_null;
    private Map<String, Double> _scoreDict_subset;
    private ChainClassifier.InferenceType _solverType;
    private Set<Chain> _predChainSet;
    private Set<Mention[]> _predSubsetPairs;
    private static final double alpha = 1;
    private Collection<Mention> _mentionSet;
    private String _docID;

    public ILPSolverThread(ChainClassifier.InferenceType type,
                           Collection<Mention> mentionSet,
                           BinaryClassifierScoreDict scoreDict_coref)
    {
        init(type, mentionSet, scoreDict_coref,
                new BinaryClassifierScoreDict(), new HashMap<>());
    }

    public ILPSolverThread(ChainClassifier.InferenceType type,
                           Collection<Mention> mentionSet,
                           BinaryClassifierScoreDict scoreDict_coref,
                           BinaryClassifierScoreDict scoreDict_null,
                           Map<String, Double> scoreDict_subset)
    {
        init(type, mentionSet, scoreDict_coref,
                scoreDict_null, scoreDict_subset);
    }

    private void init(ChainClassifier.InferenceType type,
                      Collection<Mention> mentionSet,
                      BinaryClassifierScoreDict scoreDict_coref,
                      BinaryClassifierScoreDict scoreDict_null,
                      Map<String, Double> scoreDict_subset)
    {
        //initialize everything
        _scoreDict_coref = scoreDict_coref;
        _scoreDict_subset = scoreDict_subset;
        _scoreDict_null = scoreDict_null;
        _mentionSet = mentionSet;
        _predChainSet = new HashSet<>();
        _predSubsetPairs = new HashSet<>();
        _docID = ((Mention)_mentionSet.toArray()[0]).getDocID();

        //setup the solver
        _solver = new GurobiHook();
        _solverType = type;
        _solver.setMaximize(true);
    }

    public void run()
    {
        switch (_solverType){
            case ILP_BEST_LINK:
            case ILP_ALL_LINK:
            case ILP_ALL_CONSTRAINTS: run_coref();
                break;
            case ILP_RELATION: run_full();
                break;
        }
    }

    public void run_coref()
    {
        //get an in-order list of mentions (since the solver operates
        //on indices)
        List<Mention> mentionList = new ArrayList<>(_mentionSet);

        //int[][] bestClusterIndices = new int[mentionList.size()][mentionList.size()];
        int[][] bestLinkIndices = new int[mentionList.size()][mentionList.size()];
        for(int i=0; i<mentionList.size(); i++){
            Mention m = mentionList.get(i);
            int[] idxI = new int[mentionList.size()-1];
            double[] valueI = new double[mentionList.size()-1];
            for(int j=i+1; j<mentionList.size(); j++){
                Mention mPrime = mentionList.get(j);

                //Add a boolean variable. This corresponds to
                //      w_uv * y_uv
                //where w is the confidence score of coreference between
                //mentions u and v; y is the boolean variable we're adding
                //to the ILP
                double coeff = 0;
                if(_scoreDict_coref.get(m,mPrime) != null)
                    coeff = _scoreDict_coref.get(m,mPrime);
                if(_solverType == ChainClassifier.InferenceType.ILP_ALL_CONSTRAINTS){
                    boolean sameCap = m.getCaptionIdx() == mPrime.getCaptionIdx();
                    boolean sameType = m.getLexicalType().equals(mPrime.getLexicalType());
                    if(!sameType){
                        if(m.getLexicalType().contains("other") || mPrime.getLexicalType().contains("other")) {
                            sameType = true;
                        } else {
                            String[] typeArr_1 = m.getLexicalType().split("/");
                            String[] typeArr_2 = mPrime.getLexicalType().split("/");
                            Set<String> typeSet = new HashSet<>();
                            typeSet.addAll(Arrays.asList(typeArr_1));
                            typeSet.retainAll(Arrays.asList(typeArr_2));
                            if(typeSet.size() > 1)
                                sameType = true;
                        }
                    }
                    if(sameCap)
                        coeff = 0;
                }
                bestLinkIndices[i][j] = _solver.addBooleanVariable(coeff);

                //add a cluster variable
                //bestClusterIndices[i][j] = _solver.addBooleanVariable(alpha * coeff);

                //in the original code, they could use j as the idx (or... i... in their code)
                //but our looping is different, so our idx has to be... weirder
                idxI[j-i-1] = bestLinkIndices[i][j];
                valueI[j-i-1] = 1;

                //add a consistency constraint between Best Link and Best Cluster
                //_solver.addGreaterThanConstraint(new int[]{bestClusterIndices[i][j], bestLinkIndices[i][j]}, new double[]{1.0, -1.0}, 0.0);
            }
            //Add a constraint that at mention i, we only want the _best_
            //link to mention j (so we only consider one edge at the moment)
            _solver.addLessThanConstraint(idxI, valueI, 1);

            //I'm actually pretty confident this is the transitive closure bit from
            //the all link method, which says we must enforce the following
            //      y_uw >= y_uv + y_vw - 1     \forall u,w,v
            //      1 >= y_uv + y_vw - y_uw     \forall u,w,v
            //      y_uv + y_vw - y_uw <= 1      \forall u,w,v
            //which is exactly what we're doing here!
            if(_solverType == ChainClassifier.InferenceType.ILP_ALL_LINK ||
               _solverType == ChainClassifier.InferenceType.ILP_ALL_CONSTRAINTS){
                for(int j=i+1; j<mentionList.size(); j++){
                    for(int k=j+1; k<mentionList.size(); k++){
                        _solver.addLessThanConstraint(new int[]{bestLinkIndices[j][k],
                                        bestLinkIndices[i][j], bestLinkIndices[i][k]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{bestLinkIndices[j][k],
                                        bestLinkIndices[i][j], bestLinkIndices[i][k]},
                                new double[]{1.0, -1.0, 1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{bestLinkIndices[j][k],
                                        bestLinkIndices[i][j], bestLinkIndices[i][k]},
                                new double[]{-1.0, 1.0, 1.0}, 1.0);
                    }
                }
            }
        }

        //Solve the ILP
        try {
            _solver.solve();
        } catch (Exception ex) {
            Logger.log(ex);
        }

        Map<Mention, String> mentionChainIdDict = new HashMap<>();
        int chainIdx = 0;
        for(int i=0; i<mentionList.size(); i++){
            Mention m1 = mentionList.get(i);
            String chainID_1 = mentionChainIdDict.get(m1);
            for(int j=i+1; j<mentionList.size(); j++){
                Mention m2 = mentionList.get(j);
                String chainID_2 = mentionChainIdDict.get(m2);

                //skip pairs that are already linked
                if(chainID_1 != null && chainID_2 != null && chainID_1.equals(chainID_2))
                    continue;

                //if the solver says these two should be linked...
                Boolean link = null;
                try{
                    link = _solver.getBooleanValue(bestLinkIndices[i][j]);
                } catch(Exception ex) {
                    Logger.log("getBoolean");
                    Logger.log(ex);
                }
                Double s = null;
                try{
                  s = _scoreDict_coref.get(m1, m2);
                } catch (Exception ex){
                    Logger.log("score dict");
                    Logger.log(ex);
                }


                if(link != null && link && s != null && s >= 0){
                //if(_solver.getBooleanValue(bestLinkIndices[i][j]) && _scoreDict_coref.get(m1, m2) >= 0){
                    //NOTE: intellij is giving warnings here and rightly so; I'm basically
                    //      constructing an ifelse block with an ab truth table
                    //          if a && !b
                    //          elif !a && b
                    //          elif !a && !b
                    //          elif a && b
                    //      which is equivalent to
                    //          if a && !b
                    //          elif !a && b
                    //          elif !a
                    //          else
                    //      However, I like my statements explicit, so the structure stays

                    //a) if one of the mentions has an ID already and the other
                    //   doesn't, copy the ID
                    if(chainID_1 != null && chainID_2 == null){
                        mentionChainIdDict.put(m2, chainID_1);
                    } else if (chainID_1 == null && chainID_2 != null) {
                        mentionChainIdDict.put(m1, chainID_2);
                    } //b) if neither m1 nor m2 have a chain ID, put them both in a new chain
                    else if(chainID_1 == null && chainID_2 == null){
                        mentionChainIdDict.put(m1, String.valueOf(chainIdx));
                        mentionChainIdDict.put(m2, String.valueOf(chainIdx));
                        chainIdx++;
                    } //c) if both m1 and m2 have ID's and they aren't the same, merge
                    else if(chainID_1 != null && chainID_2 != null){
                        Set<Mention> reassigMentionSet = new HashSet<>();
                        for(Mention m : mentionChainIdDict.keySet())
                            if(mentionChainIdDict.get(m).equals(chainID_2))
                                reassigMentionSet.add(m);
                        reassigMentionSet.forEach(m -> mentionChainIdDict.put(m, chainID_1));
                    }
                }
            }
        }


        //if we haven't put mention 1 in the dict at all, set it as
        //a singleton chain
        for(Mention m : mentionList){
            if(!mentionChainIdDict.containsKey(m)){
                mentionChainIdDict.put(m, String.valueOf(chainIdx));
                chainIdx++;
            }
        }

        //finally, repackage everything into actual chains
        Map<String, Set<Mention>> chainIdMentionSetDict =
                Util.invertMap(mentionChainIdDict);
        for(String chainID : chainIdMentionSetDict.keySet()) {
            Chain c = new Chain(_docID, chainID);
            chainIdMentionSetDict.get(chainID).forEach(m -> c.addMention(m));
            _predChainSet.add(c);
        }
    }

    public void run_full()
    {
        //get an in-order list of mentions (since the solver operates
        //on indices)
        List<Mention> mentionList = new ArrayList<>(_mentionSet);

        int[][] linkIndices_coref = new int[mentionList.size()][mentionList.size()];
        int[][] linkIndices_subset = new int[mentionList.size()][mentionList.size()];
        int[][] linkIndices_null = new int[mentionList.size()][mentionList.size()];
        for(int i=0; i<mentionList.size(); i++){
            Mention m = mentionList.get(i);
            List<Integer> iIndices = new ArrayList<>();

            for(int j=i+1; j<mentionList.size(); j++){
                Mention mPrime = mentionList.get(j);

                //Add four boolean variables, corresponding to
                //our 4 possible labels
                String ID_mmprime = m.getUniqueID() + "|" +
                        mPrime.getUniqueID();
                String ID_mprimem = mPrime.getUniqueID() + "|" +
                        m.getUniqueID();
                double corefScore = 0;
                if(_scoreDict_coref.get(m,mPrime) != null)
                    corefScore = _scoreDict_coref.get(m,mPrime);
                double subsetScore_1 = 0;
                double subsetScore_2 = 0;
                if(_scoreDict_subset.containsKey(ID_mmprime))
                    subsetScore_1 = _scoreDict_subset.get(ID_mmprime);
                if(_scoreDict_subset.containsKey(ID_mprimem))
                    subsetScore_2 = _scoreDict_subset.get(ID_mprimem);
                double nullScore = 0;
                if(_scoreDict_null.get(m,mPrime) != null)
                    nullScore = _scoreDict_null.get(m,mPrime);

                linkIndices_coref[i][j] = _solver.addBooleanVariable(corefScore);
                linkIndices_coref[j][i] = _solver.addBooleanVariable(corefScore);
                linkIndices_subset[i][j] = _solver.addBooleanVariable(subsetScore_1);
                linkIndices_subset[j][i] = _solver.addBooleanVariable(subsetScore_2);
                linkIndices_null[i][j] = _solver.addBooleanVariable(nullScore);
                linkIndices_null[j][i] = _solver.addBooleanVariable(nullScore);

                //Add the boolean variables to our list, such that
                //we can keep only the best label of the best edge
                iIndices.add(linkIndices_coref[i][j]);
                iIndices.add(linkIndices_subset[i][j]);
                iIndices.add(linkIndices_subset[j][i]);
                iIndices.add(linkIndices_null[i][j]);

                //Links can only take one label
                _solver.addEqualityConstraint(new int[]{linkIndices_coref[i][j],
                    linkIndices_subset[i][j], linkIndices_subset[j][i],
                    linkIndices_null[i][j]}, new double[]{1.0, 1.0, 1.0, 1.0}, 1.0);

                //Symmetry between the coref and null links
                _solver.addEqualityConstraint(new int[]{linkIndices_coref[i][j], linkIndices_coref[j][i]}, new double[]{1.0, -1.0}, 0.0);
                _solver.addEqualityConstraint(new int[]{linkIndices_null[i][j], linkIndices_null[j][i]}, new double[]{1.0, -1.0}, 0.0);

                //Transitive closure for coreference
                for(int k=j+1; k<mentionList.size(); k++){
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_coref[i][j], linkIndices_coref[i][k]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_coref[i][j], linkIndices_coref[i][k]},
                            new double[]{1.0, -1.0, 1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_coref[i][j], linkIndices_coref[i][k]},
                            new double[]{-1.0, 1.0, 1.0}, 1.0);
                }

                //Subset transitivity
                for(int k=j+1; k<mentionList.size(); k++){
                    _solver.addLessThanConstraint(new int[]{linkIndices_subset[i][k],
                        linkIndices_subset[k][j], linkIndices_subset[i][j]},
                            new double[]{1.0,1.0,-1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_subset[k][i],
                                    linkIndices_subset[j][k], linkIndices_subset[j][i]},
                            new double[]{1.0,1.0,-1.0}, 1.0);
                }

                //entity relation consistency
                for(int k=j+1; k<mentionList.size(); k++){
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                            linkIndices_subset[i][k], linkIndices_subset[i][j]},
                            new double[]{1.0,1.0,-1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                            linkIndices_subset[k][i], linkIndices_subset[j][i]},
                            new double[]{1.0,1.0,-1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_null[i][k], linkIndices_null[i][j]},
                            new double[]{1.0,1.0,-1.0}, 1.0);
                }
            }

            //We want to constrain that each time through the loop, we keep only
            //the best edge between i and j, and each edge must have only
            //one label
            //UPDATE: In the coreference case, we need this constraint because
            //        we want to keep only the best coreference links (such that they're
            //        transitively closed) and we ignore all other links, implicitly
            //        assigning a null label. Because we reason about
            //        the full graph, we can't do the same.
            int[] iIdxArr = new int[iIndices.size()];
            for(int j=0; j<iIndices.size(); j++)
                iIdxArr[j] = iIndices.get(j);
            double[] iValueArr = new double[iIndices.size()];
            Arrays.fill(iValueArr, 1.0);
            //_solver.addLessThanConstraint(iIdxArr, iValueArr, 1.0);
        }

        //Solve the ILP
        try {
            _solver.solve();
        } catch (Exception ex) {
            Logger.log(ex);
        }


        _predSubsetPairs = new HashSet<>();
        Map<Mention, String> mentionChainIdDict = new HashMap<>();
        int chainIdx = 0;
        for(int i=0; i<mentionList.size(); i++){
            Mention m1 = mentionList.get(i);
            String chainID_1 = mentionChainIdDict.get(m1);
            for(int j=i+1; j<mentionList.size(); j++){
                Mention m2 = mentionList.get(j);
                String chainID_2 = mentionChainIdDict.get(m2);

                //Get our linkv alues (only one should be non-null and
                //true)
                Boolean link_coref = null;
                try{
                    link_coref = _solver.getBooleanValue(linkIndices_coref[i][j]);
                }catch(Exception ex){/*Do nothing*/}

                Boolean link_subsetij = null;
                try{
                    link_subsetij = _solver.getBooleanValue(linkIndices_subset[i][j]);
                }catch(Exception ex){/*Do nothing*/}

                Boolean link_subsetji = null;
                try{
                    link_subsetji = _solver.getBooleanValue(linkIndices_subset[j][i]);
                }catch(Exception ex){/*Do nothing*/}

                //switch, depending on our predicted label
                if(link_coref != null && link_coref){
                    //a) if one of the mentions has an ID already and the other
                    //   doesn't, copy the ID
                    if(chainID_1 != null && chainID_2 == null){
                        mentionChainIdDict.put(m2, chainID_1);
                    } else if (chainID_1 == null && chainID_2 != null) {
                        mentionChainIdDict.put(m1, chainID_2);
                    } //b) if neither m1 nor m2 have a chain ID, put them both in a new chain
                    else if(chainID_1 == null){
                        mentionChainIdDict.put(m1, String.valueOf(chainIdx));
                        mentionChainIdDict.put(m2, String.valueOf(chainIdx));
                        chainIdx++;
                    } //c) if both m1 and m2 have ID's and they aren't the same, merge
                    else{
                        Set<Mention> reassigMentionSet = new HashSet<>();
                        for(Mention m : mentionChainIdDict.keySet())
                            if(mentionChainIdDict.get(m).equals(chainID_2))
                                reassigMentionSet.add(m);
                        reassigMentionSet.forEach(m -> mentionChainIdDict.put(m, chainID_1));
                    }
                } else if(link_subsetij != null && link_subsetij){
                    _predSubsetPairs.add(new Mention[]{m1, m2});
                } else if(link_subsetji != null && link_subsetji){
                    _predSubsetPairs.add(new Mention[]{m2,m1});
                }
            }
        }

        //if we haven't put mention 1 in the dict at all, set it as
        //a singleton chain
        for(Mention m : mentionList){
            if(!mentionChainIdDict.containsKey(m)){
                mentionChainIdDict.put(m, String.valueOf(chainIdx));
                chainIdx++;
            }
        }

        //finally, repackage everything into actual chains
        Map<String, Set<Mention>> chainIdMentionSetDict =
                Util.invertMap(mentionChainIdDict);
        for(String chainID : chainIdMentionSetDict.keySet()) {
            Chain c = new Chain(_docID, chainID);
            chainIdMentionSetDict.get(chainID).forEach(m -> c.addMention(m));
            _predChainSet.add(c);
        }
    }

    public Set<Chain> getPredictedChains(){return _predChainSet;}

    public Set<Mention[]> getPredictedSubsetPairs(){return _predSubsetPairs;}

    public String getDocID(){return _docID;}
}