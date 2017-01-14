package learn;

import edu.illinois.cs.cogcomp.lbjava.infer.GurobiHook;
import structures.Document;
import structures.Mention;
import utilities.Logger;

import java.util.*;


/**Implementation of the ILP solver used for multithreaded coreference
 * inference, which wraps the CogComp ILP solver;
 * large sections of code borrowed from [Dan's group][link].
 * [link]: https://github.com/xiaoling/wikifier/blob/master/src/edu/illinois/cs/cogcomp/lbj/coref/decoders/ILPDecoder.java
 */
public class ILPSolverThread extends Thread
{
    private edu.illinois.cs.cogcomp.lbjava.infer.ILPSolver _solver;
    private Map<String, double[]> _scoreDict;
    private Map<Mention, Map<Mention, Integer>> _graph;
    private List<Mention> _mentionList;
    private String _docID;
    private Map<String, Integer> _fixedLinks;

    public ILPSolverThread(List<Mention> mentionList,
                           Map<String, double[]> scoreDict)
    {
        _init(mentionList, scoreDict, new HashMap<>());
    }

    public ILPSolverThread(List<Mention> mentionList,
           Map<String, double[]> scoreDict,
           Map<String, Integer> fixedLinks)
    {
        _init(mentionList, scoreDict, fixedLinks);
    }

    private void _init(List<Mention> mentionList,
           Map<String, double[]> scoreDict,
           Map<String, Integer> fixedLinks)
    {
        _graph = new HashMap<>();
        _mentionList = mentionList;
        _docID = _mentionList.get(0).getDocID();
        _scoreDict = scoreDict;
        _solver = new GurobiHook();
        _solver.setMaximize(true);
        if(fixedLinks == null)
            fixedLinks = new HashMap<>();
        _fixedLinks = fixedLinks;
    }

    public void run()
    {
        int[][] linkIndices_coref = new int[_mentionList.size()][_mentionList.size()];
        int[][] linkIndices_subset = new int[_mentionList.size()][_mentionList.size()];
        int[][] linkIndices_supset = new int[_mentionList.size()][_mentionList.size()];
        int[][] linkIndices_null = new int[_mentionList.size()][_mentionList.size()];
        for(int i=0; i<_mentionList.size(); i++){
            Mention m_i = _mentionList.get(i);

            for(int j=i+1; j<_mentionList.size(); j++){
                Mention m_j = _mentionList.get(j);

                //Add boolean variables for each label, each direction
                String id_ij = Document.getMentionPairStr(m_i, m_j, true, true);
                String id_ji = Document.getMentionPairStr(m_j, m_i, true, true);
                double score_coref_ij = 0.0, score_coref_ji = 0.0;
                double score_subset_ij = 0.0, score_subset_ji = 0.0;
                double score_supset_ij = 0.0, score_supset_ji = 0.0;
                double score_null_ij = 0.0, score_null_ji = 0.0;

                //if(_scoreDict.containsKey(ID_ij) && Mention.getLexicalTypeMatch(m_i, m_j) > 0){
                if(_scoreDict.containsKey(id_ij)){
                    score_null_ij = _scoreDict.get(id_ij)[0];
                    score_coref_ij = _scoreDict.get(id_ij)[1];
                    score_subset_ij = _scoreDict.get(id_ij)[2];
                    score_supset_ij = _scoreDict.get(id_ij)[3];
                }
                //if(_scoreDict.containsKey(ID_ji) && Mention.getLexicalTypeMatch(m_i, m_j) > 0){
                if(_scoreDict.containsKey(id_ji)){
                    score_null_ji = _scoreDict.get(id_ji)[0];
                    score_coref_ji = _scoreDict.get(id_ji)[1];
                    score_subset_ji = _scoreDict.get(id_ji)[2];
                    score_supset_ji = _scoreDict.get(id_ji)[3];
                }

                linkIndices_null[i][j] = _solver.addBooleanVariable(score_null_ij);
                linkIndices_null[j][i] = _solver.addBooleanVariable(score_null_ji);
                linkIndices_coref[i][j] = _solver.addBooleanVariable(score_coref_ij);
                linkIndices_coref[j][i] = _solver.addBooleanVariable(score_coref_ji);
                linkIndices_subset[i][j] = _solver.addBooleanVariable(score_subset_ij);
                linkIndices_subset[j][i] = _solver.addBooleanVariable(score_subset_ji);
                linkIndices_supset[i][j] = _solver.addBooleanVariable(score_supset_ij);
                linkIndices_supset[j][i] = _solver.addBooleanVariable(score_supset_ji);

                //If either link is in our set of fixed links, constrain
                //the ILP accordingly
                if(_fixedLinks.containsKey(id_ij)){
                    double[] coeff = {0.0, 0.0, 0.0, 0.0};
                    coeff[_fixedLinks.get(id_ij)] = 1.0;
                    _solver.addEqualityConstraint(new int[]{linkIndices_null[i][j],
                            linkIndices_coref[i][j], linkIndices_subset[i][j], linkIndices_supset[i][j]},
                            coeff, 1.0);
                }
                if(_fixedLinks.containsKey(id_ji)){
                    double[] coeff = {0.0, 0.0, 0.0, 0.0};
                    coeff[_fixedLinks.get(id_ji)] = 1.0;
                    _solver.addEqualityConstraint(new int[]{linkIndices_null[j][i],
                                    linkIndices_coref[j][i], linkIndices_subset[j][i], linkIndices_supset[j][i]},
                            coeff, 1.0);
                }

                //Links can only take one label in each direction
                _solver.addEqualityConstraint(new int[]{linkIndices_null[i][j],
                    linkIndices_coref[i][j], linkIndices_subset[i][j], linkIndices_supset[i][j]},
                    new double[]{1.0, 1.0, 1.0, 1.0}, 1.0);
                _solver.addEqualityConstraint(new int[]{linkIndices_null[j][i],
                        linkIndices_coref[j][i], linkIndices_subset[j][i], linkIndices_supset[j][i]},
                        new double[]{1.0, 1.0, 1.0, 1.0}, 1.0);

                //Enforce symmetry between coref / antisym between subset
                _solver.addEqualityConstraint(new int[]{linkIndices_coref[i][j], linkIndices_coref[j][i]}, new double[]{1.0, -1.0}, 0.0);
                _solver.addEqualityConstraint(new int[]{linkIndices_subset[i][j], linkIndices_supset[j][i]}, new double[]{1.0, -1.0}, 0.0);
                _solver.addEqualityConstraint(new int[]{linkIndices_supset[i][j], linkIndices_subset[j][i]}, new double[]{1.0, -1.0}, 0.0);

                //subset transitivity
                for(int k=j+1; k<_mentionList.size(); k++){
                    _solver.addLessThanConstraint(new int[]{linkIndices_subset[i][k],
                                    linkIndices_subset[k][j], linkIndices_subset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_subset[j][k],
                                    linkIndices_subset[k][i], linkIndices_subset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);

                    _solver.addLessThanConstraint(new int[]{linkIndices_supset[i][k],
                                    linkIndices_supset[k][j], linkIndices_supset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_supset[j][k],
                                    linkIndices_supset[k][i], linkIndices_supset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                }

                //entity relation consistency
                for(int k=j+1; k<_mentionList.size(); k++){
                    //null link consistency
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_null[k][j], linkIndices_null[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //ik entity egress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_null[j][k], linkIndices_null[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //ik entity ingress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_null[k][i], linkIndices_null[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //jk entity egress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_null[i][k], linkIndices_null[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //jk entity ingress

                    //coref transitive closure consistency
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_coref[k][j], linkIndices_coref[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_coref[k][i], linkIndices_coref[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);

                    //subset consistency
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_subset[k][j], linkIndices_subset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_subset[j][k], linkIndices_subset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_subset[k][i], linkIndices_subset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_subset[i][k], linkIndices_subset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0);

                    //superset consistency
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_supset[k][j], linkIndices_supset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //ik entity egress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[i][k],
                                    linkIndices_supset[j][k], linkIndices_supset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //ik entity ingress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_supset[k][i], linkIndices_supset[j][i]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //jk entity egress
                    _solver.addLessThanConstraint(new int[]{linkIndices_coref[j][k],
                                    linkIndices_supset[i][k], linkIndices_supset[i][j]},
                            new double[]{1.0, 1.0, -1.0}, 1.0); //jk entity ingress

                }
            }
        }

        //Solve the ILP
        try {
            _solver.solve();
        } catch (Exception ex) {
            Logger.log(ex);
        }

        //Store the complete graph
        for(int i=0; i<_mentionList.size(); i++){
            Mention m1 = _mentionList.get(i);

            for(int j=i+1; j<_mentionList.size(); j++){
                Mention m2 = _mentionList.get(j);

                //get and store the ij link
                Integer link_ij = null;
                if(_solver.getBooleanValue(linkIndices_null[i][j]))
                    link_ij = 0;
                else if(_solver.getBooleanValue(linkIndices_coref[i][j]))
                    link_ij = 1;
                else if(_solver.getBooleanValue(linkIndices_subset[i][j]))
                    link_ij = 2;
                else if(_solver.getBooleanValue(linkIndices_supset[i][j]))
                    link_ij = 3;

                if(!_graph.containsKey(m1))
                    _graph.put(m1, new HashMap<>());
                _graph.get(m1).put(m2, link_ij);

                //get and store the ji link
                Integer link_ji = null;
                if(_solver.getBooleanValue(linkIndices_null[j][i]))
                    link_ji = 0;
                else if(_solver.getBooleanValue(linkIndices_coref[j][i]))
                    link_ji = 1;
                else if(_solver.getBooleanValue(linkIndices_subset[j][i]))
                    link_ji = 2;
                else if(_solver.getBooleanValue(linkIndices_supset[j][i]))
                    link_ji = 3;
                if(!_graph.containsKey(m2))
                    _graph.put(m2, new HashMap<>());
                _graph.get(m2).put(m1, link_ji);
            }
        }
    }

    public Map<Mention, Map<Mention, Integer>> getRelationGraph(){return _graph;}

    public String getDocID(){return _docID;}
}