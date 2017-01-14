package core;

import learn.*;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.*;
import net.sourceforge.argparse4j.internal.HelpScreenException;
import net.sourceforge.argparse4j.internal.UnrecognizedArgumentException;
import nlptools.WordnetUtil;
import out.OutTable;
import statistical.ScoreDict;
import structures.*;
import utilities.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;


/**The Overlord is responsible for
 * parsing args and calling other PanOpt
 * modules
 * 
 * @author ccervantes
 */
public class Overlord 
{
    public static String dataPath = "/home/ccervan2/source/data/";
    public static String lexPath = "/shared/projects/DenotationGraphGeneration/data/lexiconsNew/";
    public static String captionTePath = "/shared/projects/caption_te/";
    public static String wordnetDir = "/shared/data/WordNet-3.0/dict/";
    public static String word2vecPath = "/shared/projects/word2vec/word2vec.vector.gz";
    public static String datasetPath = "/shared/projects/Flickr30kEntities_v2/";
    public static String datasetPath_legacy = "/shared/projects/Flickr30kEntities/";
    public static String dbPath = datasetPath + "Flickr30kEntities_v2.db";
    public static String resourcesDir = datasetPath + "resources/";
    public static String dbPath_legacy = datasetPath_legacy + "Flickr30kEntities_v1.db";
    public static String boxFeatureDir = dataPath + "Flickr30kEntities_v1/box_feats/";
    public static String boxMentionDir = dataPath + "Flickr30kEntities_v2/box_mention_embeddings/";

	private static String _outroot;

	/**Main function. Parses <b>args</b> and calls
	 * other PanOpt modules
	 * 
	 * @param args - Command line arguments
	 */
	public static void main(String[] args)
	{
        //debug();
	    //System.exit(0);

        //start our runtime clock
		Logger.startClock();

		//arguments for the  manager
		ArgumentParser parser = 
				ArgumentParsers.newArgumentParser("ImageCaptionLearn");
		parser.defaultHelp(true);
		parser.description("ImageCaptionLearn has several " +
                "modules, each with their own args. To get " +
                "more info on a particular module, specify " +
                "the module with --help");
        setArgument_flag(parser.addArgument("--quiet"),
                "Whether to log output");
        setArgument(parser.addArgument("--log_delay"),
                "Minimum seconds to wait between logging progress status messages",
                Integer.class, 90);
        setArgument(parser.addArgument("--out"),
                "Writes output to file with ROOT prefix", "ROOT");
        setArgument(parser.addArgument("--threads"),
                "Uses NUM threads, where applicable",
                Integer.class, 1, "NUM", false);
        String[] dataOpts = {"train", "dev", "test", "all"};
        setArgument_opts(parser.addArgument("--data"), dataOpts, null,
                "Where applicable, loads only the specified data");

        //Throwaway arguments will appear in the Debug module;
        //More permanant things will appear in their own modules
        Subparsers subparsers = parser.addSubparsers();
        Subparser debugParser = subparsers.addParser("Debug");
        setArgument_flag(debugParser.addArgument("--heterog_subset"),
                "Exports a list of heterog subset pairs");
        setArgument_flag(debugParser.addArgument("--penult_filter"),
                "Exports mention pairs for which the penult filter fires");
        setArgument_flag(debugParser.addArgument("--mod_subset"),
                "Exports newly modified subset features");

        /* Data Group */
        Subparser dataParser = subparsers.addParser("Data");
        setArgument(dataParser.addArgument("--convert_to_arff"),
                "Converts the specified .feats file to .arff format",
                "PATH");
        String[] featOpts = {"pairwise", "affinity", "nonvis"};
        setArgument_opts(dataParser.addArgument("--extractFeats"), featOpts, null,
                "Extracts features to --out");

        /* Learn Group */
        Subparser learnParser = subparsers.addParser("Learn");
        setArgument(learnParser.addArgument("--train_file"), "Training data", "PATH");
        setArgument(learnParser.addArgument("--model_file"), "Model file (save to / load from)", "PATH");
        setArgument_flag(learnParser.addArgument("--inf"), "Pairwise identity inference");
        String[] learners = {"weka_multi", "liblinear_logistic"};
        setArgument_opts(learnParser.addArgument("--learner"), learners, "weka_multi",
                "Specifies which training is taking place");
        setArgument(learnParser.addArgument("--batch_size"), "train arg; uses SIZE batches",
                Integer.class, 100, "SIZE", false);
        setArgument(learnParser.addArgument("--epochs"), "train arg; run for NUM epochs",
                Integer.class, 1000, "NUM", false);
        setArgument(learnParser.addArgument("--eval_file"), "Evaluation data", "PATH");
        setArgument_flag(learnParser.addArgument("--pronom_coref"),
                "Evaluates rule-based pronominal coreference resolution");
        setArgument(learnParser.addArgument("--nonvis_scores"), "inf arg; nonvisual scores", "FILE");
        setArgument(learnParser.addArgument("--pairwise_scores"), "inf arg; pairwise identity scores", "FILE");

        //Actually parse the arguments
        Namespace ns = parseArgs(parser, args);

		//parse our main args
		if(!ns.getBoolean("quiet"))
			Logger.setVerbose();
		_outroot = ns.getString("out");
        int numThreads = ns.getInt("threads");
        Logger.setStatusDelay(ns.getInt("log_delay"));
        String dataSplit = ns.getString("data");
        Collection<Document> docSet = null;
        if(dataSplit != null){
            DBConnector conn = new DBConnector(dbPath);
            if(dataSplit.equals("all"))
                docSet = DocumentLoader.getDocumentSet(conn);
            else if(dataSplit.equals("dev"))
                docSet = DocumentLoader.getDocumentSet(conn, 0);
            else if(dataSplit.equals("train"))
                docSet = DocumentLoader.getDocumentSet(conn, 1);
            else if(dataSplit.equals("test"))
                docSet = DocumentLoader.getDocumentSet(conn, 2);
        }

        //Switch on the specified module, and parse module args
        List<String> argList = Arrays.asList(args);
        if(argList.contains("Debug")){
            if(ns.getBoolean("heterog_subset")) {
                Minion.export_subsetHeterogType(docSet);
            } else if(ns.getBoolean("penult_filter")){
                Minion.export_penultFilter(docSet);
            } else if(ns.getBoolean("mod_subset")){
                Minion.export_modSubsetFeats(docSet, dataSplit);
            } else {

                DoubleDict<Integer> labelHist = new DoubleDict<>();
                for(Document d : docSet){
                    Set<String> subsetPairs = d.getSubsetMentions();

                    List<Mention> mentions = d.getMentionList();
                    for(int i=0; i<mentions.size(); i++){
                        Mention m_i = mentions.get(i);
                        if(m_i.getChainID().equals("0"))
                            continue;
                        if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_i.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            continue;

                        for(int j=i+1; j<mentions.size(); j++){
                            Mention m_j = mentions.get(j);

                            if(m_j.getChainID().equals("0"))
                                continue;
                            if(m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                               m_j.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                                continue;

                            String id_ij = Document.getMentionPairStr(m_i, m_j, true, true);
                            String id_ji = Document.getMentionPairStr(m_j, m_i, true, true);

                            int label_ij = 0, label_ji = 0;
                            if(m_i.getChainID().equals(m_j.getChainID())){
                                label_ij = 1; label_ji = 1;
                            }

                            if(subsetPairs.contains(id_ij))
                                label_ij = 2;
                            if(subsetPairs.contains(id_ji))
                                label_ji = 2;

                            if(label_ij == 2 && label_ji == 2){
                                System.out.println("------");
                                System.out.println(m_i.toString() + "|" + m_j.toString());
                                System.out.println(d.getCaption(m_i.getCaptionIdx()).toEntitiesString());
                                System.out.println(d.getCaption(m_j.getCaptionIdx()).toEntitiesString());
                            }

                            if(label_ij == 2)
                                label_ji = 3;
                            else if(label_ji == 2)
                                label_ij = 3;

                            labelHist.increment(label_ij);
                            labelHist.increment(label_ji);
                        }
                    }
                }

                System.out.println(labelHist);

                System.exit(0);



                List<String> ll = FileIO.readFile_lineList(Overlord.dataPath + "feats/flickr30kEntities_v2_pairwise_dev.feats");
                Map<String, Integer> labelDict = new HashMap<>();
                for(String line : ll){
                    FeatureVector fv = FeatureVector.parseFeatureVector(line);
                    labelDict.put(fv.comments, (int)fv.label);
                }
                for(Document d : docSet){
                    List<Mention> mentions = d.getMentionList();
                    for(int i=0; i<mentions.size(); i++){
                        Mention m_i = mentions.get(i);
                        if(m_i.getChainID().equals("0"))
                            continue;

                        if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_i.getPronounType() != Mention.PRONOUN_TYPE.SEMI){
                            continue;
                        }

                        for(int j=i+1; j<mentions.size(); j++){
                            Mention m_j = mentions.get(j);
                            if(m_j.getChainID().equals("0"))
                                continue;

                            if(m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                               m_j.getPronounType() != Mention.PRONOUN_TYPE.SEMI){
                                continue;
                            }

                            String id_ij = Document.getMentionPairStr(m_i, m_j, true, true);
                            String id_ji = Document.getMentionPairStr(m_j, m_i, true, true);

                            if(!labelDict.containsKey(id_ij) || !labelDict.containsKey(id_ji)){
                                System.out.println("-------");
                                System.out.println(id_ij);
                                System.out.println(m_i.toString() + " | " + m_j.toString());
                                System.out.println(labelDict.containsKey(id_ij));
                                System.out.println(labelDict.containsKey(id_ji));
                            }

                            if(labelDict.get(id_ij) == 2 && labelDict.get(id_ji) != 3 ||
                               labelDict.get(id_ij) == 3 && labelDict.get(id_ji) != 2){
                                System.out.println("-------");
                                System.out.println(id_ij);
                                System.out.println(m_i.toString() + " | " + m_j.toString());
                                System.out.println(labelDict.get(id_ij) + " | " + labelDict.get(id_ji));
                                Set<String> boxes_i = new HashSet<>();
                                for(BoundingBox b : d.getBoxSetForMention(m_i))
                                    boxes_i.add(""+b.getIdx());
                                Set<String> boxes_j = new HashSet<>();
                                for(BoundingBox b : d.getBoxSetForMention(m_j))
                                    boxes_j.add(""+b.getIdx());
                                System.out.println(StringUtil.listToString(boxes_i, "|"));
                                System.out.println(StringUtil.listToString(boxes_j, "|"));
                                System.out.println(d.getCaption(m_i.getCaptionIdx()).toEntitiesString());
                                System.out.println(d.getCaption(m_j.getCaptionIdx()).toEntitiesString());
                            }
                        }
                    }
                }

                System.exit(0);


                Minion.export_filteredMentionPairCases(docSet,
                        Minion :: filter_boxSubsetCases, "ex_subsetCases");
                System.exit(0);



                Set<String> hypernyms = new HashSet<>();
                for(String[] row : FileIO.readFile_table(Overlord.resourcesDir + "hist_hypernym.csv"))
                    hypernyms.add(row[0]);

                Map<String, Object> hd = new HashMap<>();
                WordnetUtil wn = new WordnetUtil(Overlord.wordnetDir);
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        Set<String> leaves = new HashSet<>();
                        HypTree tree = wn.getHypernymTree(m.getHead().getLemma());
                        for(List<HypTree.HypNode> branch : tree.getRootBranches()) {
                            String leaf = null;
                            for (HypTree.HypNode node : branch) {
                                if(hypernyms.contains(node.toString())) {
                                    leaf = node.toString();
                                    break;
                                }
                            }
                            if(leaf != null)
                                leaves.add(leaf);
                        }
                        hd.put(m.getUniqueID(), leaves);
                    }
                }
                JsonIO.writeFile(hd, "id_hyp_dict");
                System.exit(0);

                Logger.log("Mysql");
                Minion.buildImageCaptionDB(Overlord.datasetPath + "Flickr30kEntities_v2.coref",
                        Overlord.datasetPath + "RELEASE/", Overlord.resourcesDir + "img_comments.csv",
                        Overlord.resourcesDir + "img_crossval.csv", "engr-cpanel-mysql.engr.illinois.edu",
                        "ccervan2_root", "thenIdefyheaven!", "ccervan2_imageCaption");
                Logger.log("SqlLite");
                Minion.buildImageCaptionDB(Overlord.datasetPath + "Flickr30kEntities_v2.coref",
                        Overlord.datasetPath + "RELEASE/", Overlord.resourcesDir + "img_comments.csv",
                        Overlord.resourcesDir + "img_crossval.csv", Overlord.dbPath.replace(".db", "_new.db"));
                System.exit(0);

                List<String> ll_lems = new ArrayList<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        ll_lems.add(m.getUniqueID() + "," + m.getHead().getLemma().toLowerCase().replace(",", "[COMMA]"));
                    }
                }
                FileIO.writeFile(ll_lems, "ex_lemma", "csv");
                System.exit(0);


                WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
                Set<String> _hypernyms = new HashSet<>();
                for(String[] row : FileIO.readFile_table(Overlord.resourcesDir + "hist_hypernym.csv"))
                    _hypernyms.add(row[0]);

                Map<String, Set<String>> hypDict = new HashMap<>();
                DoubleDict<String> lemmaHist = new DoubleDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        String lemma = m.getHead().getLemma().toLowerCase();
                        lemmaHist.increment(lemma);
                        if(!hypDict.containsKey(lemma)){
                            Set<String> leaves = new HashSet<>();
                            HypTree tree = wnUtil.getHypernymTree(m.getHead().getLemma());
                            for(List<HypTree.HypNode> branch : tree.getRootBranches()) {
                                String leaf = null;
                                for (HypTree.HypNode node : branch) {
                                    if(_hypernyms.contains(node.toString())) {
                                        leaf = node.toString();
                                        break;
                                    }
                                }
                                if(leaf != null)
                                    leaves.add(leaf);
                            }
                            hypDict.put(lemma, leaves);
                        }
                    }
                }

                List<String> ll_hyps = new ArrayList<>();
                for(String lem : hypDict.keySet())
                    ll_hyps.add(lem + "," + StringUtil.listToString(hypDict.get(lem), "|") + "," + lemmaHist.get(lem));
                FileIO.writeFile(ll_hyps, "hist_hypernym_lemmas", "csv", false);
                System.exit(0);

                Map<String, DoubleDict<String>> hypVisHist = new HashMap<>();
                for(String hyp : _hypernyms)
                    hypVisHist.put(hyp, new DoubleDict<>());

                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        Set<String> leaves = new HashSet<>();
                        HypTree tree = wnUtil.getHypernymTree(m.getHead().getLemma().toLowerCase());
                        for(List<HypTree.HypNode> branch : tree.getRootBranches()){
                            String leaf = null;
                            for(HypTree.HypNode node : branch){
                                if(_hypernyms.contains(node.toString())){
                                    leaf = node.toString();
                                    break;
                                }
                            }
                            if(leaf != null)
                                leaves.add(leaf);
                        }
                        for(String leaf : leaves){
                            if(m.getChainID().equals("0"))
                                hypVisHist.get(leaf).increment("nonvis");
                            else
                                hypVisHist.get(leaf).increment("vis");
                        }
                    }
                }

                OutTable ot = new OutTable("hyp", "vis_count", "vis_%", "nonvis_count", "nonvis_%");
                double total_vis = 0, total_nonvis = 0;
                for(String hyp : hypVisHist.keySet()){
                    total_vis += hypVisHist.get(hyp).get("vis");
                    total_nonvis += hypVisHist.get(hyp).get("nonvis");
                }
                for(String hyp : hypVisHist.keySet()){
                    int vis = (int)hypVisHist.get(hyp).get("vis");
                    int nonvis = (int)hypVisHist.get(hyp).get("nonvis");
                    ot.addRow(hyp, vis, 100.0 * vis / total_vis, nonvis, 100.0 * nonvis / total_nonvis);
                }
                ot.writeToCsv("hist_hyps", true);

                System.exit(0);
            }
        } else if(argList.contains("Data")) {
            String featsFileToConvert = ns.getString("convert_to_arff");
            String featsToExtract = ns.getString("extractFeats");

            if(featsFileToConvert != null) {
                WekaMulticlass.exportToArff(featsFileToConvert);
            } else if(featsToExtract != null){
                if(featsToExtract.equals("pairwise"))
                    ClassifyUtil.exportFeatures_pairwise(docSet, _outroot, numThreads);
                else if(featsToExtract.equals("affinity"))
                    ClassifyUtil.export_affinityFeats(docSet, dataSplit);
                else if(featsToExtract.equals("nonvis")) {
                    ClassifyUtil.initLists();
                    ClassifyUtil.exportFeatures_nonvis(docSet, _outroot);
                }
            }

        } else if(argList.contains("Learn")) {
            Logger.log("Reading scores files");
            List<String> ll_pairwise_scores = FileIO.readFile_lineList(ns.getString("pairwise_scores"));
            Map<String, double[]> pairwise_scoreDict = new HashMap<>();
            for(String line : ll_pairwise_scores){
                String[] lineParts = line.split(",");
                double[] scores = new double[4];
                for(int i=1; i<lineParts.length; i++)
                    scores[i-1] = Math.exp(Double.parseDouble(lineParts[i]));
                pairwise_scoreDict.put(lineParts[0], scores);
            }

            BinaryClassifierScoreDict nonvis_scoreDict = new BinaryClassifierScoreDict(ns.getString("nonvis_scores"));
            RelationInference.infer(docSet, pairwise_scoreDict, numThreads, nonvis_scoreDict);
            Logger.log("Evaluation");
            ScoreDict<Integer> graphScores = RelationInference.evaluateGraph();
            graphScores.printConfusionMatrix();
            for(int label : graphScores.keySet())
                System.out.printf("%d & %s & %d (%.2f%%)\n", label, graphScores.getScore(label).toLatexString(),
                        graphScores.getGoldCount(label),
                        100.0 * (double)graphScores.getGoldCount(label) / graphScores.getTotalGold());
            Logger.log("Accuracy");
            for(int label : graphScores.keySet())
                System.out.printf("%d : %.2f%%\n", label, graphScores.getAccuracy(label));
            System.out.printf("total : %.2f%%\n", graphScores.getAccuracy());
            Logger.log("Pairwise confusion");
            RelationInference.printPairwiseConfusion();

            Logger.log("Exporting conll files to out/coref/conll/");
            Map<String, Set<Chain>> docChainSetDict = RelationInference.getPredictedChains();
            for(Document d : docSet){
                List<String> lineList_key = d.toConll2012();
                lineList_key.add(0, "#begin document (" + d.getID() + "); part 000");
                lineList_key.add("#end document");
                FileIO.writeFile(lineList_key, "out/coref/conll/" + d.getID().replace(".jpg", "") + "_key", "conll", false);

                Set<Chain> predChainSet = docChainSetDict.get(d.getID());
                if(predChainSet == null){
                    Logger.log("Document %s has no predicted chains", d.getID());
                    predChainSet = new HashSet<>();
                }
                List<String> lineList_resp = Document.toConll2012(d, predChainSet);
                lineList_resp.add(0, "#begin document (" + d.getID() + "); part 000");
                lineList_resp.add("#end document");
                FileIO.writeFile(lineList_resp, "out/coref/conll/" + d.getID().replace(".jpg", "") + "_response", "conll", false);
            }

            Logger.log("Exporting htm files");
            Map<String, Set<Chain[]>> predSubsetChains = RelationInference.getPredictedSubsetChains();
            System.out.println("Docs with pred subsets: " + predSubsetChains.keySet().size());
            int numDocsWithGoldSubsets = 0;
            for(Document d : docSet)
                if(!d.getSubsetMentions().isEmpty())
                    numDocsWithGoldSubsets++;
            System.out.println("Docs with gold subsets: " + numDocsWithGoldSubsets);
            List<String> docIds = new ArrayList<>(predSubsetChains.keySet());
            Collections.shuffle(docIds);
            for(int i=0; i<100; i++){
                Document d = null;
                for(Document dPrime : docSet)
                    if(dPrime.getID().equals(docIds.get(i)))
                        d = dPrime;
                if(d != null){
                    FileIO.writeFile(HtmlIO.getImgHtm(d, docChainSetDict.get(d.getID()),
                            predSubsetChains.get(d.getID())),
                            "out/coref/htm/" + d.getID().replace(".jpg", ""),
                            "htm", false);
                }
            }
        }
	}

    /**Returns the Namespace object for the given parser, run over
     * the given args; Quits the application if arg parser is violated
     *
     * @param parser
     * @param args
     * @return
     */
    private static Namespace parseArgs(ArgumentParser parser, String[] args)
    {
        //actually parse our args
        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch(ArgumentParserException apEx) {
            //Only print help if the exception wasn't a help
            //exception (I guess those are printed
            //automatically)
            if(!apEx.getClass().equals(HelpScreenException.class)) {
                parser.printHelp();
                if(apEx.getClass().equals(UnrecognizedArgumentException.class)) {
                    UnrecognizedArgumentException unArgEx =
                            (UnrecognizedArgumentException)apEx;
                    System.out.println("\n***ERROR*** Unrecognized argument: " +
                            unArgEx.getArgument());
                }
                else if(apEx.getMessage().contains("ambiguous option") ||
                        apEx.getMessage().contains("invalid choice") ||
                        apEx.getMessage().contains("is required") ||
                        apEx.getMessage().contains("unrecognized arguments") ||
                        apEx.getMessage().contains("too few arguments"))
                {
                    System.out.println("\n***ERROR*** " + apEx.getMessage());
                } else {
                    System.out.println(apEx.getMessage());
                    apEx.printStackTrace();
                }
            }
            System.exit(0);
        }
        return ns;
    }

    /**Sets up an flag argument, which stores true if specified
     *
     * @param arg
     * @param help
     */
    private static void setArgument_flag(Argument arg, String help)
    {
        arg.help(help);
        arg.action(Arguments.storeTrue());
    }

    /**Sets up an option argument, allowing for one of opts options
     *
     * @param arg
     * @param opts
     * @param defaultVal
     * @param help
     */
    private static void setArgument_opts(Argument arg, String[] opts,
                                         String defaultVal, String help)
    {
        arg.choices(Arrays.asList(opts));
        if(defaultVal != null)
            arg.setDefault(defaultVal);
        arg.help(help);
    }

    /**Sets up a string argument with the specified help text
     *
     * @param arg
     * @param help
     */
    private static void setArgument(Argument arg, String help)
    {
        setArgument(arg, help, null, null, null, null);
    }

    /**Sets up an argument of type with defaultVal and help
     *
     * @param arg
     * @param help
     * @param type
     * @param defaultVal
     */
    private static void setArgument(Argument arg, String help,
                                    Class type, Object defaultVal)
    {
        setArgument(arg, help, type, defaultVal, null, null);
    }

    /**Sets up a string argument with specified help and meta-var
     *
     * @param arg
     * @param help
     * @param metavar
     */
    private static void setArgument(Argument arg, String help, String metavar)
    {
        setArgument(arg, help, null, null, metavar, null);
    }

    /**Sets up a new argument with specified options
     *
     * @param arg
     * @param help
     * @param type
     * @param defaultVal
     * @param metavar
     * @param required
     */
    private static void setArgument(Argument arg, String help,
                                    Class type, Object defaultVal, String metavar,
                                    Boolean required)
    {
        arg.help(help);
        if(metavar != null)
            arg.metavar(metavar);
        if(defaultVal != null)
            arg.setDefault(defaultVal);
        if(type != null)
            arg.type(type);
        if(required != null)
            arg.required(required);
    }

    public static boolean getConsoleConfirm() {
        System.out.print("Are you sure you want to continue (y/n)? ");
        char c = ' ';
        BufferedReader br =
                new BufferedReader(new InputStreamReader(System.in));
        try{
            String line = br.readLine();
            if(line.length() == 1)
                c = line.toCharArray()[0];
            while(c != 'y' && c != 'Y' && c != 'n' && c != 'N'){
                System.out.print("y or n, please: ");
                line = br.readLine();
                if(line.length() == 1)
                    c = line.toCharArray()[0];
            }
        } catch (IOException ioEx) {
            Logger.log(ioEx);
        }
        return c == 'y' || c == 'Y';
    }

}
