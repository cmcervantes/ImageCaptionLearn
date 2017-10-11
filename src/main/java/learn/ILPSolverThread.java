package learn;

import edu.illinois.cs.cogcomp.lbjava.infer.GurobiHook;
import gurobi.GRB;
import gurobi.GRBEnv;
import gurobi.GRBException;
import structures.BoundingBox;
import structures.Document;
import structures.Mention;
import utilities.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**Implementation of the ILP solver used for multithreaded coreference
 * inference, which wraps the CogComp ILP solver;
 * large sections of code borrowed from [Dan's group][link].
 * [link]: https://github.com/xiaoling/wikifier/blob/master/src/edu/illinois/cs/cogcomp/lbj/coref/decoders/ILPDecoder.java
 */
public class ILPSolverThread extends Thread {

    private String _docID;
    private List<Mention> _mentionList;
    private List<BoundingBox> _boxList;
    private ILPInference.InferenceType _infType;
    private int _solverThreads;

    private Map<String, Integer> _fixedRelationLinks;
    private Map<String, double[]> _relScores, _cardScores;
    private Map<String, Double> _affScores;
    private Map<String, Integer> _relationGraph, _groundingGraph;

    private edu.illinois.cs.cogcomp.lbjava.infer.ILPSolver _solver;
    private boolean _foundSolution;
    private boolean _fallbackSolution;

    private boolean _includeSubset;
    private boolean _includeTypeConstraint;
    private boolean _includeBoxExigence;
    private int _maxRelationLabel;

    private Map<Mention, String> _mentionCatDict;

    /**Constructor for relation inference
     *
     * @param mentionList
     * @param solverThreads
     */
    public ILPSolverThread(List<Mention> mentionList, int solverThreads)
    {
        _init(mentionList, null, ILPInference.InferenceType.RELATION, solverThreads);
    }

    /**Constructor for grounding and joint inference
     *
     * @param mentionList
     * @param boxList
     * @param infType
     * @param solverThreads
     */
    public ILPSolverThread(List<Mention> mentionList, List<BoundingBox> boxList,
                           ILPInference.InferenceType infType, int solverThreads)
    {
        _init(mentionList, boxList, infType, solverThreads);
    }

    /**Initializes all the appropriate variables; assumes various sets
     * are going to be called before a thread is actually executed
     *
     * @param mentionList
     * @param boxList
     * @param infType
     * @param solverThreads
     */
    private void _init(List<Mention> mentionList, List<BoundingBox> boxList,
                       ILPInference.InferenceType infType, int solverThreads)
    {
        _mentionList = mentionList;
        _docID = _mentionList.get(0).getDocID();
        if(boxList != null)
            _boxList = boxList;
        _infType = infType;
        _solverThreads = solverThreads;

        _fixedRelationLinks = new HashMap<>(); _relScores = new HashMap<>();
        _cardScores = new HashMap<>(); _affScores = new HashMap<>();

        _includeSubset = true;
        _includeTypeConstraint = false;
        _includeBoxExigence = true;
        _maxRelationLabel = 3;
        _mentionCatDict = new HashMap<>();

        //set up our graphs and solver
        _relationGraph = new HashMap<>(); _groundingGraph = new HashMap<>();
        _foundSolution = false; _fallbackSolution = false;
        _solverThreads = solverThreads;
        _resetSolver();
    }

    /**Sets up the solver; must be run again for
     * settings like relation-after-grounding
     */
    private void _resetSolver()
    {
        GRBEnv gurobiEnv = null;
        try {
            gurobiEnv = new GRBEnv("gurobi.log");
            gurobiEnv.set(GRB.IntParam.Threads, _solverThreads);
            gurobiEnv.set(GRB.IntParam.OutputFlag, 0);
            gurobiEnv.set(GRB.DoubleParam.TimeLimit, 600);
        } catch(GRBException grEx){
            Logger.log(grEx);
        }
        _solver = new GurobiHook(gurobiEnv);
        _solver.setMaximize(true);
    }

    /**
     * Returns the docID with which this thread is associated
     *
     * @return
     */
    public String getDocID() {
        return _docID;
    }

    /**Whether ILP inference found a solution
     *
     * @return
     */
    public boolean foundSolution(){return _foundSolution;}

    /**Whether ILP inference had to fall back to a solution
     *
     * @return
     */
    public boolean isFallbackSolution(){return _fallbackSolution;}


    /* Setup Methods */

    /**Sets the fixed relation links to use during relation
     * or joint inference
     *
     * @param fixedRelationLinks
     */
    public void setFixedRelationLinks(Map<String, Integer> fixedRelationLinks)
    {
        _fixedRelationLinks = new HashMap<>(fixedRelationLinks);
    }

    /**Specifies that during relation or joint inference
     * subset relations should be excluded
     */
    public void excludeSubset()
    {
        _includeSubset = false;
        _maxRelationLabel = 1;
    }

    /**Specifies that a type constraint should be used during
     * inference (which inference -- and which type constraint --
     * has changed over multiple revisions)
     *
     */
    public void includeTypeConstraint()
    {
        _includeTypeConstraint = true;

        //Now that we know the type constraint
        //is active, look up all our mentions'
        //COCO categories
        for(Mention m : _mentionList){
            String cocoCat = Mention.getLexicalEntry_cocoCategory(m, true);
            if(cocoCat != null)
                _mentionCatDict.put(m, cocoCat);
        }
    }

    /**WSpecifies that the box exigence constraint should
     * be excluded (because datasets like MSCOCO are noisier
     * than ours)
     *
     */
    public void excludeBoxExigence()
    {
        _includeBoxExigence = false;
    }

    /**Adds the relation scores to the solver thread
     *
     * @param relationScores
     */
    public void setRelationScores(Map<String, double[]> relationScores)
    {
        _relScores = relationScores;
    }

    /**Adds the cardinality scores to the solver thread
     *
     * @param cardinalityScores
     */
    public void setCardinalityScores(Map<String, double[]> cardinalityScores)
    {
        _cardScores = cardinalityScores;
    }

    /**Adds the affinity scores to the solver thread
     *
     * @param affinityScores
     */
    public void setAffinityScores(Map<String, double[]> affinityScores)
    {
        affinityScores.forEach((k, v) -> _affScores.put(k, affinityScores.get(k)[1]));
    }


    /* Run Methods */

    /**
     * Sets up and runs the ILP solver based on the
     * internal inference type
     */
    public void run() {
        switch (_infType) {
            case RELATION:
                run_relation();
                break;
            case GROUNDING:
                run_grounding();
                break;
            case JOINT:
                run_joint();
                break;
            case JOINT_AFTER_REL:
                run_relation();
                _fixedRelationLinks = new HashMap<>(_relationGraph);
                _relationGraph = new HashMap<>();
                run_joint();
                break;
            case JOINT_AFTER_GRND:
                run_grounding();
                _fixedRelationLinks = new HashMap<>(_groundingGraph);
                _groundingGraph = new HashMap<>();
                run_joint();
                break;
        }

        //If we failed to find a joint solution, find individual solutions
        if(_infType.toString().contains("JOINT") && !_foundSolution){
            _fallbackSolution = true;
            _relationGraph = new HashMap<>(); _groundingGraph = new HashMap<>();
            _resetSolver();
            run_relation();
            boolean solvedRelation = _foundSolution;
            _resetSolver();
            run_grounding();
            _foundSolution &= solvedRelation;
        }
    }

    /**
     * Sets up and runs the ILP solver for relation inference
     */
    private void run_relation() {
        int[][][] relationIndices = new int[_mentionList.size()][_mentionList.size()][_maxRelationLabel +1];
        for (int i = 0; i < _mentionList.size(); i++) {
            Mention m_i = _mentionList.get(i);
            for (int j = i + 1; j < _mentionList.size(); j++) {
                Mention m_j = _mentionList.get(j);

                //Add boolean variables for each label, each direction
                String id_ij = Document.getMentionPairStr(m_i, m_j);
                String id_ji = Document.getMentionPairStr(m_j, m_i);
                for (int y = 0; y <= _maxRelationLabel; y++) {
                    relationIndices[i][j][y] = _addRelationVariable(id_ij, y, 1.0);
                    relationIndices[j][i][y] = _addRelationVariable(id_ji, y, 1.0);
                }

                //Add pairwise relation constraints
                _addRelationConstraints_pairwise(relationIndices[i][j], relationIndices[j][i]);
            }
        }

        //Add fixed links
        _addRelationConstraints_fixed(relationIndices);

        //Add subset transitivity / entity consistency
        _addRelationConstraints_transitivity(relationIndices);

        //Solve the ILP
        solveGraph(relationIndices, null);
    }

    /**
     * Sets up and runs the ILP solver for grounding inference
     */
    private void run_grounding() {
        int[][] groundingIndices = new int[_mentionList.size()][_boxList.size()];
        int[][] antiGroundingIndices = new int[_mentionList.size()][_boxList.size()];
        for (int i = 0; i < _mentionList.size(); i++) {
            Mention m_i = _mentionList.get(i);
            int[] groundingIndices_perMention = new int[_boxList.size()];
            for (int o = 0; o < _boxList.size(); o++) {
                BoundingBox b_g = _boxList.get(o);
                String id_io = m_i.getUniqueID() + "|" + b_g.getUniqueID();
                groundingIndices[i][o] = _addGroundingVariable_affinity(id_io);
                antiGroundingIndices[i][o] = _addGroundingVariable_antiAffinity(id_io,
                        groundingIndices[i][o]);
                groundingIndices_perMention[o] = groundingIndices[i][o];
            }

            //Add the cardinality variables
            int[] cardinalityIndices_perMention = _addGroundingVariable_cardinality(m_i.getUniqueID());
            _addGroundingConstraint_cardinality(groundingIndices_perMention, cardinalityIndices_perMention);
        }

        //Given that we know these are gold boxes, we must
        //assign a box to at least one mention
        if(_includeBoxExigence)
            _addGroundingConstraint_boxExigence(groundingIndices);

        if(_includeTypeConstraint)
            _addGroundingConstraints_category(groundingIndices);

        //Solve the ILP
        solveGraph(null, groundingIndices);
    }

    /**
     * Sets up and runs the ILP solver for combined inference
     */
    private void run_joint() {
        int[][][] relationIndices = new int[_mentionList.size()][_mentionList.size()][_maxRelationLabel +1];
        int[][] groundingIndices = new int[_mentionList.size()][_boxList.size()];
        int[][] antiGroundingIndices = new int[_mentionList.size()][_boxList.size()];
        int[][] cardinalityIndices = new int[_mentionList.size()][_boxList.size() + 1];
        for (int i = 0; i < _mentionList.size(); i++) {
            Mention m_i = _mentionList.get(i);
            for (int j = i + 1; j < _mentionList.size(); j++) {
                Mention m_j = _mentionList.get(j);

                //Add boolean variables for each label, each direction
                String id_ij = Document.getMentionPairStr(m_i, m_j);
                String id_ji = Document.getMentionPairStr(m_j, m_i);
                for (int y = 0; y <= _maxRelationLabel; y++) {
                    relationIndices[i][j][y] = _addRelationVariable(id_ij, y,
                            2.0 / _mentionList.size());
                    relationIndices[j][i][y] = _addRelationVariable(id_ji, y,
                            2.0 / _mentionList.size());
                }

                //Add pairwise relation constraints
                _addRelationConstraints_pairwise(relationIndices[i][j], relationIndices[j][i]);
            }

            int[] groundingIndices_perMention = new int[_boxList.size()];
            for (int o = 0; o < _boxList.size(); o++) {
                BoundingBox b_g = _boxList.get(o);
                String id_io = m_i.getUniqueID() + "|" + b_g.getUniqueID();
                groundingIndices[i][o] = _addGroundingVariable_affinity(id_io);
                antiGroundingIndices[i][o] = _addGroundingVariable_antiAffinity(id_io,
                        groundingIndices[i][o]);
                groundingIndices_perMention[o] = groundingIndices[i][o];
            }

            //Add the cardinality variables
            cardinalityIndices[i] = _addGroundingVariable_cardinality(m_i.getUniqueID());
            _addGroundingConstraint_cardinality(groundingIndices_perMention, cardinalityIndices[i]);
        }

        //Add grounded relation constraints
        _addJointConstraints(relationIndices, groundingIndices, cardinalityIndices);

        //Add fixed links
        _addRelationConstraints_fixed(relationIndices);

        //Add subset transitivity / entity consistency
        _addRelationConstraints_transitivity(relationIndices);

        //Given that we know these are gold boxes, we must
        //assign a box to at least one mention
        if(_includeBoxExigence)
            _addGroundingConstraint_boxExigence(groundingIndices);

        //Solve the ILP
        solveGraph(relationIndices, groundingIndices);
    }

    /**Calls the solver to solve the ILP graph and stores the graph(s)
     *
     * @param relationIndices
     * @param groundingIndices
     */
    private void solveGraph(int[][][] relationIndices, int[][] groundingIndices)
    {
        try {
            _foundSolution = _solver.solve();
        } catch (Exception ex) {
            Logger.log(ex);
        }

        if(_foundSolution) {
            //Store the relation graph
            if(relationIndices != null)
                _saveRelationGraph(relationIndices);

            //Store the grounding graph
            if(groundingIndices != null)
                _saveGroundingGraph(groundingIndices);
        }
    }

    /* Variable Methods */

    /**Adds a boolean relation variable to the solver, given the mention pair's ID,
     * the label the variable is supposed to represent, the heterogeneous type cost,
     * and the coefficient
     *
     * @param pairID    - Mention pair ID of the end points to this variable's link
     * @param label     - [0,3] label of the link this variable represents
     * @param coeff     - coefficient of this score (typically 1.0 or 2/|M|)
     * @return
     */
    private int _addRelationVariable(String pairID, int label, double coeff)
    {
        double score = 0.0;
        if (_relScores.containsKey(pairID))
            score = Math.max(0, coeff * _relScores.get(pairID)[label]);
        return _solver.addBooleanVariable(score);
    }

    /**Adds a boolean grounding variable, returning the
     * variable's index; given pairID specifies the
     * affinity score to look up when setting the
     * boolean variable's coefficient
     *
     * @param pairID
     * @return
     */
    private int _addGroundingVariable_affinity(String pairID)
    {
        double score = 0.0;
        if (_affScores.containsKey(pairID))
            score = _affScores.get(pairID);
        score /= (double)_boxList.size();
        return _solver.addBooleanVariable(score);
    }

    /**Adds the anti-affinity variable, which is 1 when the affinity
     * variable is 0 and 0 when affinity is 1
     *
     * @param pairID
     * @param affinityIdx
     * @return
     */
    private int _addGroundingVariable_antiAffinity(String pairID, int affinityIdx)
    {
        double score = 0.0;
        if (_affScores.containsKey(pairID))
            score = 1 - _affScores.get(pairID);
        score /= (double)_boxList.size();

        //Add the anti-affinity variable and constrain it to be on only
        //when affinity is off
        int antiAffinityIdx = _solver.addBooleanVariable(score);
        _solver.addEqualityConstraint(new int[]{affinityIdx, antiAffinityIdx},
                new double[]{1.0, 1.0}, 1.0);
        return antiAffinityIdx;
    }

    /**Adds the cardinality variable, which is one only when a mention
     * is ground to that many boxes
     *
     * @param mentionID
     * @return
     */
    private int[] _addGroundingVariable_cardinality(String mentionID)
    {
        int[] indices = new int[_boxList.size() + 1];
        for(int n=0; n<=_boxList.size(); n++){
            double cardScore = 0.0;
            if(_cardScores.containsKey(mentionID)){
                if(n < 11)
                    cardScore = _cardScores.get(mentionID)[n];
                else
                    cardScore = _cardScores.get(mentionID)[11] / (_boxList.size() - 10.0);
            }
            indices[n] = _solver.addBooleanVariable(cardScore);
        }
        return indices;
    }

    /* Constraint Methods */

    /**Adds the grounded coreference and grounded subset constraints to the solver
     *
     * @param relationIndices
     * @param groundingIndices
     */
    private void _addJointConstraints(int[][][] relationIndices, int[][] groundingIndices, int[][] cardinalityIndices)
    {
        double beta = 2 * _boxList.size() + 1;
        for (int i = 0; i < _mentionList.size(); i++) {
            for (int j = i + 1; j < _mentionList.size(); j++) {
                for(int o=0; o<_boxList.size(); o++){
                    //If i and j are coreferent, they must share eactly the same boxes
                    _solver.addLessThanConstraint(new int[]{relationIndices[i][j][1],
                        groundingIndices[i][o], groundingIndices[j][o]},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{relationIndices[j][i][1],
                        groundingIndices[j][o], groundingIndices[i][o]},
                        new double[]{1.0, 1.0, -1.0}, 1.0);

                    //If i subset j, j must have at least the boxes in i
                    _solver.addLessThanConstraint(new int[]{relationIndices[i][j][2],
                        groundingIndices[i][o], groundingIndices[j][o]},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                    _solver.addLessThanConstraint(new int[]{relationIndices[j][i][2],
                        groundingIndices[j][o], groundingIndices[i][o]},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                }

                /* Subsets must be strictly smaller than their superset but only in cases where
                 * either i or j are both associated with boxes; otherwise we don't care about
                 * groundings */

                //Store whether either i or j has boxes; note that v_ij is symmetric
                // -1 <= z^0_i + z^0_j - 2v_ij <= 0
                //v_ij is only 0 when both z^0_i and z^0_j are 0
                int card_0_i = cardinalityIndices[i][0], card_0_j = cardinalityIndices[j][0];
                int v_ij = _solver.addBooleanVariable(0.0);
                _solver.addGreaterThanConstraint(new int[]{card_0_i, card_0_j, v_ij},
                        new double[]{1.0, 1.0, 2.0}, 2.0);
                _solver.addLessThanConstraint(new int[]{card_0_i, card_0_j, v_ij},
                        new double[]{1.0, 1.0, 2.0}, 3.0);

                //Store whether i is less than j or j is less than i
                // 0 <= \sum_o g_{io} - \sum_o g_{jo} + \beta w_ij <= \beta - 1
                // 0 <= \sum_o g_{jo} - \sum_o g_{io} + \beta w_ji <= \beta - 1
                //w_ij: i < j; w_ji: j < i
                int[] groundingIndices_i = groundingIndices[i], groundingIndices_j = groundingIndices[j];
                int[] subsetIndices_ij = new int[2 * _boxList.size()+1];
                int[] subsetIndices_ji = new int[2 * _boxList.size()+1];
                int w_ij = _solver.addBooleanVariable(0.0), w_ji = _solver.addBooleanVariable(0.0);
                subsetIndices_ij[0] = w_ij; subsetIndices_ji[0] = w_ji;
                System.arraycopy(groundingIndices_i, 0, subsetIndices_ij, 1, _boxList.size());
                System.arraycopy(groundingIndices_j, 0, subsetIndices_ij, _boxList.size()+1, _boxList.size());
                System.arraycopy(groundingIndices_j, 0, subsetIndices_ji, 1, _boxList.size());
                System.arraycopy(groundingIndices_i, 0, subsetIndices_ji, _boxList.size()+1, _boxList.size());
                double[] subsetCoeffs = new double[2 * _boxList.size()+1];
                subsetCoeffs[0] = beta;
                Arrays.fill(subsetCoeffs, 1, _boxList.size()+1, 1.0);
                Arrays.fill(subsetCoeffs, _boxList.size()+1, 2*_boxList.size()+1, -1.0);
                _solver.addGreaterThanConstraint(subsetIndices_ij, subsetCoeffs, 0);
                _solver.addLessThanConstraint(subsetIndices_ij, subsetCoeffs, beta - 1);
                _solver.addGreaterThanConstraint(subsetIndices_ji, subsetCoeffs, 0);
                _solver.addLessThanConstraint(subsetIndices_ji, subsetCoeffs, beta - 1);

                _solver.addLessThanConstraint(new int[]{relationIndices[i][j][2], v_ij, w_ij},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                _solver.addLessThanConstraint(new int[]{relationIndices[i][j][2], w_ij, v_ij},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                _solver.addLessThanConstraint(new int[]{relationIndices[j][i][2], v_ij, w_ji},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
                _solver.addLessThanConstraint(new int[]{relationIndices[j][i][2], w_ji, v_ij},
                        new double[]{1.0, 1.0, -1.0}, 1.0);
            }
        }
    }

    /**Adds the relation transitivity constraints; includes
     * sub/superset transitivity and entity consistency (implicitly including
     * coref transitive closure)
     *
     * @param linkIndices
     */
    private void _addRelationConstraints_transitivity(int[][][] linkIndices)
    {
        //Iterate through all the links (assuming we have all our variables set up)
        //and set the transitivity / consistency constraints
        for(int i=0; i<_mentionList.size(); i++){
            Mention m_i = _mentionList.get(i);
            for(int j=i+1; j<_mentionList.size(); j++){
                Mention m_j = _mentionList.get(j);
                for(int k=0; k<_mentionList.size(); k++){
                    //We consider the ij / ji pair along with each mention
                    //k with which ij could have a relation
                    if(k == i || k == j)
                        continue;
                    Mention m_k = _mentionList.get(k);

                    /* Subset Transitivity */
                    //If there exists an ij subset link, any subset link to/from k
                    //must hold for both i and j
                    if(_includeSubset){
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][2],
                                        linkIndices[j][k][2], linkIndices[i][k][2]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[j][i][2],
                                        linkIndices[i][k][2], linkIndices[j][k][2]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][2],
                                        linkIndices[k][i][2], linkIndices[k][j][2]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[j][i][2],
                                        linkIndices[k][j][2], linkIndices[k][i][2]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                    }

                    /* Entity Relation Consistency */
                    for(int y = 0; y<= _maxRelationLabel; y++){
                        //If there exists an ij coref link, any link to/from k
                        //must hold for both i and j
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][1],
                                        linkIndices[i][k][y], linkIndices[j][k][y]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][1],
                                        linkIndices[j][k][y], linkIndices[i][k][y]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][1],
                                        linkIndices[k][i][y], linkIndices[k][j][y]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][1],
                                        linkIndices[k][j][y], linkIndices[k][i][y]},
                                new double[]{1.0, 1.0, -1.0}, 1.0);
                    }

                    /* Type constraints for subsets */
                    if(_includeTypeConstraint){
                        double typeOrPronom_jk = m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                                m_k.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                                Mention.getLexicalTypeMatch(m_j, m_k) > 0 ? 1.0 : 0.0;
                        double typeOrPronom_ik = m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                                m_k.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                                Mention.getLexicalTypeMatch(m_i, m_k) > 0 ? 1.0 : 0.0;

                        _solver.addLessThanConstraint(new int[]{linkIndices[i][j][2], linkIndices[i][k][2]},
                                new double[]{1.0, 1.0}, 1.0 + typeOrPronom_jk);

                        _solver.addLessThanConstraint(new int[]{linkIndices[j][i][2], linkIndices[j][k][2]},
                                new double[]{1.0, 1.0}, 1.0 + typeOrPronom_ik);
                    }
                }
            }
        }
    }

    /**Adds the fixed link set as relation constraints, requiring that
     * certain links take a fixed label
     *
     * @param linkIndices
     */
    private void _addRelationConstraints_fixed(int[][][] linkIndices)
    {
        //Don't bother going through the loops if we have no fixed links
        //to add
        if(_fixedRelationLinks.isEmpty())
           return;

        for(int i=0; i<_mentionList.size(); i++) {
            Mention m_i = _mentionList.get(i);
            for (int j = i + 1; j < _mentionList.size(); j++) {
                Mention m_j = _mentionList.get(j);

                String id_ij = Document.getMentionPairStr(m_i, m_j);
                String id_ji = Document.getMentionPairStr(m_j, m_i);

                if (_fixedRelationLinks.containsKey(id_ij)) {
                    double[] coeff = {0.0, 0.0, 0.0, 0.0};
                    coeff[_fixedRelationLinks.get(id_ij)] = 1.0;
                    _solver.addEqualityConstraint(linkIndices[i][j], coeff, 1.0);
                }
                if (_fixedRelationLinks.containsKey(id_ji)) {
                    double[] coeff = {0.0, 0.0, 0.0, 0.0};
                    coeff[_fixedRelationLinks.get(id_ji)] = 1.0;
                    _solver.addEqualityConstraint(linkIndices[j][i], coeff, 1.0);
                }
            }
        }
    }


    /**Adds the pairwise relation constraints: if ij is coref, so must ji;
     * if ij is sub, ji must be sup; if ij is sup, ji must be sub
     *
     * @param linkIndices_ij
     * @param linkIndices_ji
     */
    private void _addRelationConstraints_pairwise(int[] linkIndices_ij, int[] linkIndices_ji)
    {
        //Links can only take one label in each direction
        _solver.addEqualityConstraint(linkIndices_ij,
                new double[]{1.0, 1.0, 1.0, 1.0}, 1.0);
        _solver.addEqualityConstraint(linkIndices_ji,
                new double[]{1.0, 1.0, 1.0, 1.0}, 1.0);

        //Enforce symmetry between coref / antisym between subset
        _solver.addEqualityConstraint(new int[]{linkIndices_ij[1],
                linkIndices_ji[1]}, new double[]{1.0, -1.0}, 0.0);
        if(_includeSubset){
            _solver.addEqualityConstraint(new int[]{linkIndices_ij[2],
                    linkIndices_ji[3]}, new double[]{1.0, -1.0}, 0.0);
            _solver.addEqualityConstraint(new int[]{linkIndices_ij[3],
                    linkIndices_ji[2]}, new double[]{1.0, -1.0}, 0.0);
        }
    }

    /**Adds the box exigence grounding constraint, which requires that each
     * box must be associated with at least one mention
     *
     * @param linkIndices
     */
    private void _addGroundingConstraint_boxExigence(int[][] linkIndices)
    {
        for(int o=0; o<_boxList.size(); o++){
            int[] linkIndices_exigence = new int[_mentionList.size()];
            for(int i=0; i<_mentionList.size(); i++)
                linkIndices_exigence[i] = linkIndices[i][o];
            double[] coeffs_exigence = new double[_mentionList.size()];
            Arrays.fill(coeffs_exigence, 1.0);
            _solver.addGreaterThanConstraint(linkIndices_exigence, coeffs_exigence, 1.0);
        }
    }

    /**Adds the cardinality grounding constraints, which in effect enable us to have
     * a boolean variable for the current number of boxes, allowing us to add
     * a given cardinality confience to the objective
     *
     * @param groundingLinks_perMention
     * @param cardinalityLinks_perMention
     */
    private void _addGroundingConstraint_cardinality(int[] groundingLinks_perMention,
                                                     int[] cardinalityLinks_perMention)
    {
        for(int n=0; n<=_boxList.size(); n++){
            /*Mention cardinality Constraints*/
            //a) a^n_i stores if n > \sum_o g_{io}
            //   n \leq \beta a^n_i + \sum_o g_{io} \leq \beta + n - 1
            //b) b^n_i stores if \sum_o g_{io} > n
            //   -n \leq \beta b^n_i - \sum_o g_{io} \leq \beta - n - 1
            //c) z^n_i stores if a^n_i and b^n_i are both false
            //  -2 \leq -a^n_i - b^n_i - 2z^n_i \leq -1
            double beta = 2 * _boxList.size() + 1;
            int aLink = _solver.addBooleanVariable(0.0);
            int bLink = _solver.addBooleanVariable(0.0);
            int[] indices_a = new int[_boxList.size() + 1];
            int[] indices_b = new int[_boxList.size() + 1];
            indices_a[0] = aLink; indices_b[0] = bLink;
            for(int i=0; i<groundingLinks_perMention.length; i++) {
                indices_a[i + 1] = groundingLinks_perMention[i];
                indices_b[i + 1] = groundingLinks_perMention[i];
            }
            double[] coeffs_a = new double[_boxList.size() + 1];
            double[] coeffs_b = new double[_boxList.size() + 1];
            Arrays.fill(coeffs_a, 1.0); Arrays.fill(coeffs_b, -1.0);
            coeffs_a[0] = beta; coeffs_b[0] = beta;
            _solver.addGreaterThanConstraint(indices_a, coeffs_a, n);
            _solver.addLessThanConstraint(indices_a, coeffs_a, beta + n - 1);
            _solver.addGreaterThanConstraint(indices_b, coeffs_b, -n);
            _solver.addLessThanConstraint(indices_b, coeffs_b, beta - n - 1);

            //Now we know that if a and b are off, z must be on
            _solver.addGreaterThanConstraint(new int[]{aLink, bLink, cardinalityLinks_perMention[n]},
                    new double[]{-1.0, -1.0, -2.0}, -2.0);
            _solver.addLessThanConstraint(new int[]{aLink, bLink, cardinalityLinks_perMention[n]},
                    new double[]{-1.0, -1.0, -2.0}, -1.0);
        }

        //There can only be a single cardinality variable on at any time
        //NOTE: I think the other constraints should take care of this, but
        //to be safe...
        double[] coeffs_z = new double[_boxList.size() + 1];
        Arrays.fill(coeffs_z, 1.0);
        _solver.addEqualityConstraint(cardinalityLinks_perMention, coeffs_z, 1.0);
    }

    private void _addGroundingConstraints_category(int[][] groundingIndices)
    {
        for(int i=0; i<_mentionList.size(); i++){
            Mention m_i = _mentionList.get(i);
            for(int o=0; o<_boxList.size(); o++){
                BoundingBox b_o = _boxList.get(o);
                if(!_mentionCatDict.containsKey(m_i) ||
                   !_mentionCatDict.get(m_i).contains(b_o.getCategory()))
                    _solver.addEqualityConstraint(new int[]{groundingIndices[i][o]},
                            new double[]{1.0}, 0.0);
            }
        }
    }

    /* Graph methods */

    /**Stores the solver's boolean variables -- represented as
     * the given linkIndices -- as a relation graph
     *
     * @param linkIndices
     */
    private void _saveRelationGraph(int[][][] linkIndices)
    {
        for(int i=0; i<_mentionList.size(); i++){
            Mention m_i = _mentionList.get(i);
            for(int j=i+1; j<_mentionList.size(); j++){
                Mention m_j = _mentionList.get(j);
                String id_ij = Document.getMentionPairStr(m_i, m_j);
                String id_ji = Document.getMentionPairStr(m_j, m_i);

                //get and store the ij link
                Integer link_ij = null, link_ji = null;
                for(int y = 0; y<= _maxRelationLabel; y++){
                    if(_solver.getBooleanValue(linkIndices[i][j][y])){
                        if(link_ij != null)
                            System.out.printf("Found dual labels! %d | %d\n", link_ij, y);
                        link_ij = y;
                    }
                    if(_solver.getBooleanValue(linkIndices[j][i][y])){
                        if(link_ji != null)
                            System.out.printf("Found dual labels! %d | %d\n", link_ji, y);
                        link_ji = y;
                    }
                }

                _relationGraph.put(id_ij, link_ij);
                _relationGraph.put(id_ji, link_ji);
            }
        }
    }

    /**Stores the solver's boolean variables -- represented
     * as the given linkIndices -- as a grounding graph
     *
     * @param linkIndices
     */
    private void _saveGroundingGraph(int[][] linkIndices)
    {
        for(int i=0; i<_mentionList.size(); i++){
            Mention m_i = _mentionList.get(i);
            for(int o=0; o<_boxList.size(); o++){
                BoundingBox b_o = _boxList.get(o);
                String id_io = m_i.getUniqueID() + "|" + b_o.getUniqueID();
                Integer link_io = _solver.getBooleanValue(linkIndices[i][o]) ? 1 : 0;
                _groundingGraph.put(id_io, link_io);
            }
        }
    }

    /**Returns the relation graph produced by relation or joint inference
     *
     * @return
     */
    public Map<String, Integer> getRelationGraph(){return _relationGraph;}

    /**Returns the grounding graph produced by grounding or joint inference
     *
     * @return
     */
    public Map<String, Integer> getGroundingGraph(){return _groundingGraph;}
}