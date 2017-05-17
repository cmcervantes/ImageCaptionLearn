package core;

import learn.BinaryClassifierScoreDict;
import learn.ClassifyUtil;
import learn.ILPInference;
import learn.WekaMulticlass;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.*;
import net.sourceforge.argparse4j.internal.HelpScreenException;
import net.sourceforge.argparse4j.internal.UnrecognizedArgumentException;
import nlptools.IllinoisChunker;
import nlptools.IllinoisTagger;
import nlptools.WordnetUtil;
import out.OutTable;
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
    //Paths
    public static String dataPath = "/home/ccervan2/source/data/";
    public static String lexPath = "/shared/projects/DenotationGraphGeneration/data/lexiconsNew/";
    public static String captionTePath = "/shared/projects/caption_te/";
    public static String wordnetDir = "/shared/data/WordNet-3.0/dict/";
    public static String word2vecPath = "/shared/projects/word2vec/word2vec.vector.gz";
    public static String boxFeatureDir = dataPath + "Flickr30kEntities_v1/box_feats/";
    public static String boxMentionDir = dataPath + "Flickr30kEntities_v2/box_mention_embeddings/";

    //Dataset Paths
    public static String flickr30kPath = "/shared/projects/Flickr30kEntities_v2/";
    public static String flickr30kPath_legacy = "/shared/projects/Flickr30kEntities/";
    public static String[] flickr30k_mysqlParams =
            {"ccervan2.web.engr.illinois.edu", "ccervan2_root", "thenIdefyheaven!", "ccervan2_imageCaption"};
    public static String flickr30k_sqlite = flickr30kPath + "Flickr30kEntities_v2_20170209.db";
    public static String flickr30k_sqlite_legacy = flickr30kPath_legacy + "Flickr30kEntities_v1.db";
    public static String flickr30kResources = flickr30kPath + "resources/";
    public static String mscocoPath = "/shared/projects/MSCOCO/";
    public static String[] mscoco_mysqlParams =
            {"ccervan2.web.engr.illinois.edu", "ccervan2_root", "thenIdefyheaven!", "ccervan2_coco"};
    public static String mscoco_sqlite = mscocoPath + "COCO_20170418.db";
    public static String mscocoResources = mscocoPath + "resources/";
    public static String snliPath = "/shared/projects/SNLI/";

	private static String _outroot;

    private static void _debug()
    {
        DBConnector conn = new DBConnector("engr-cpanel-mysql.engr.illinois.edu",
                "ccervan2_root", "thenIdefyheaven!", "ccervan2_imageCaption");
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 0, 100);
        BinaryClassifierScoreDict nonvisScores =
                new BinaryClassifierScoreDict("/Users/syphonnihil/source/working/flickr30kEntities_v2_nonvis_dev.scores");
        for(Document d : docSet){
            Set<String> pred_nonvis = new HashSet<>();
            Set<String> gold_nonvis = new HashSet<>();
            for(Mention m : d.getMentionList()){
                if(nonvisScores.get(m) != null && nonvisScores.get(m) > 0){
                    pred_nonvis.add(m.toString());
                }
                if(m.getChainID().equals("0")){
                    gold_nonvis.add(m.toString());
                }
            }

            if(!gold_nonvis.isEmpty()){
                System.out.println(d.getID());
                System.out.println("\t pred: " + StringUtil.listToString(pred_nonvis, "|"));
                System.out.println("\t gold: " + StringUtil.listToString(gold_nonvis, "|"));
            }
        }

        System.exit(0);
        for(Document d : docSet){
            Set<Mention> visualPronoms_gold = new HashSet<>();
            for(Mention m : d.getMentionList()){
                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.SEMI &&
                   !m.getChainID().equals("0")){
                    visualPronoms_gold.add(m);
                }
            }

            Set<Mention> visualPronoms_pred = new HashSet<>();
            Set<String> predCorefPairs = ClassifyUtil.pronominalCoref(d, new HashSet<>(d.getMentionList()));
            List<Mention> mentions = d.getMentionList();
            for(int i=0; i<mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                for (int j = i + 1; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);

                    if(predCorefPairs.contains(id_ij) || predCorefPairs.contains(id_ji)){
                        if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_i.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            visualPronoms_pred.add(m_i);
                        if(m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                           m_j.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            visualPronoms_pred.add(m_j);
                    }
                }
            }

            Set<Mention> intersect = new HashSet<>(visualPronoms_gold);
            intersect.retainAll(visualPronoms_pred);

            if(!visualPronoms_gold.isEmpty() && !visualPronoms_pred.isEmpty()){
                if(visualPronoms_gold.size() > intersect.size()){
                    System.out.println("Gold not pred");
                    for(Mention m : visualPronoms_gold)
                        if(!intersect.contains(m))
                            System.out.println(m.toString() + " | " + m.toDebugString() + " | " + m.getPronounType().toString());
                }

                if(visualPronoms_pred.size() > intersect.size()){
                    System.out.println("Pred not Gold");
                    for(Mention m : visualPronoms_pred)
                        if(!intersect.contains(m))
                            System.out.println(m.toString() + " | " + m.toDebugString() + " | " + m.getPronounType().toString());
                }
            }
        }

    }

	/**Main function. Parses <b>args</b> and calls
	 * other PanOpt modules
	 * 
	 * @param args - Command line arguments
	 */
	public static void main(String[] args)
	{
	    /*
        _debug();
	    System.exit(0);*/

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
        _setArgument_flag(parser.addArgument("--quiet"),
                "Whether to log output");
        _setArgument(parser.addArgument("--log_delay"),
                "Minimum seconds to wait between logging progress status messages",
                Integer.class, 90);
        _setArgument(parser.addArgument("--out"),
                "Writes output to file with ROOT prefix", "ROOT");
        _setArgument(parser.addArgument("--threads"),
                "Uses NUM threads, where applicable",
                Integer.class, 1, "NUM", false);
        String[] datasetOpts = {"flickr30k", "mscoco"};
        _setArgument_opts(parser.addArgument("--dataset"), datasetOpts, "flickr30k",
                "Uses the specified dataset");
        String[] dataSplitOpts = {"train", "dev", "test", "all"};
        _setArgument_opts(parser.addArgument("--split"), dataSplitOpts, null,
                "Loads data of the specified cross validation split (or all)");
        _setArgument(parser.addArgument("--rand_docs"),
                "Loads NUM random documents of the given dataset and cross val split",
                Integer.class, null, "NUM", false);
        _setArgument_flag(parser.addArgument("--local"), "Runs locally; queries mysqlDB");

        //Throwaway arguments will appear in the Debug module;
        //More permanant things will appear in their own modules
        Subparsers subparsers = parser.addSubparsers();
        Subparser debugParser = subparsers.addParser("Debug");
        _setArgument_flag(debugParser.addArgument("--heterog_subset"),
                "Exports a list of heterog subset pairs");
        _setArgument_flag(debugParser.addArgument("--penult_filter"),
                "Exports mention pairs for which the penult filter fires");
        _setArgument_flag(debugParser.addArgument("--mod_subset"),
                "Exports newly modified subset features");

        /* Data Group */
        Subparser dataParser = subparsers.addParser("Data");
        _setArgument(dataParser.addArgument("--convert_to_arff"),
                "Converts the specified .feats file to .arff format",
                "PATH");
        String[] featOpts = {"relation", "affinity", "nonvis", "box_card"};
        _setArgument_opts(dataParser.addArgument("--extractFeats"), featOpts, null,
                "Extracts features to --out");
        _setArgument_opts(dataParser.addArgument("--buildDB"), new String[]{"mysql", "sqlite"}, "mysql",
                "Rebuilds the specified database in the default locations");

        /* Learn Group */
        Subparser learnParser = subparsers.addParser("Learn");
        _setArgument(learnParser.addArgument("--train_file"), "Training data", "PATH");
        _setArgument(learnParser.addArgument("--model_file"), "Model file (save to / load from)", "PATH");
        String[] learners = {"weka_multi", "liblinear_logistic"};
        _setArgument_opts(learnParser.addArgument("--learner"), learners, "weka_multi",
                "Specifies which training is taking place");
        _setArgument(learnParser.addArgument("--batch_size"), "train arg; uses SIZE batches",
                Integer.class, 100, "SIZE", false);
        _setArgument(learnParser.addArgument("--epochs"), "train arg; run for NUM epochs",
                Integer.class, 1000, "NUM", false);
        _setArgument(learnParser.addArgument("--eval_file"), "Evaluation data", "PATH");
        _setArgument_flag(learnParser.addArgument("--pronom_coref"),
                "Evaluates rule-based pronominal coreference resolution");

        /* Infer Group */
        Subparser infParser = subparsers.addParser("Infer");
        _setArgument(infParser.addArgument("--nonvis_scores"),
                "Nonvisual scores file; associates mention unique IDs with [0,1] nonvis prediction",
                "FILE");
        _setArgument(infParser.addArgument("--relation_scores"),
                "Pairwise relation scores file; associates mention pair IDs with [0,1] (n,c,b,p) predictions",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_relation_dev.scores", "FILE",
                false);
        _setArgument(infParser.addArgument("--affinity_scores"),
                "Affinity scores file; associates mention|box unique IDs with [0,1] affinity prediction",
                String.class, "/home/ccervan2/source/data/feats/flickr30kEntities_v2_affinity_dev.scores",
                "FILE", false);
        _setArgument(infParser.addArgument("--cardinality_scores"),
                "Cardinality scores file; associates mention unique IDs with [0,1] "+
                "(0-10,11+) cardinality prediction", String.class,
                "/home/ccervan2/source/data/feats/flickr30kEntities_v2_box_card_dev.scores",
                "FILE", false);
        _setArgument_opts(infParser.addArgument("--inf_type"), new String[]{"relation", "grounding", "joint"},
                "relation", "Specified which inference module to use");
        _setArgument_flag(infParser.addArgument("--type_constraint"), "Enables the inference type constraint");
        _setArgument_flag(infParser.addArgument("--export_files"), "Writes examples to out/coref/htm/ and conll "+
                "files to out/coref/conll/");
        _setArgument(infParser.addArgument("--graph_root"), "Loads previous inference "+
                "graphs from ROOT_relation.obj and ROOT_grounding.obj", "ROOT");
        _setArgument(infParser.addArgument("--alpha"), "[0,1] alpha value", Double.class, 0.0, "NUM", false);

        //Actually parse the arguments
        Namespace ns = _parseArgs(parser, args);

		//parse our main args
		if(!ns.getBoolean("quiet"))
			Logger.setVerbose();
        boolean runLocal = ns.getBoolean("local");
		_outroot = ns.getString("out");
        int numThreads = ns.getInt("threads");
        Logger.setStatusDelay(ns.getInt("log_delay"));
        String dataset = ns.getString("dataset");
        String split = ns.getString("split");
        Collection<Document> docSet = null;
        if(split != null){
            DBConnector conn = null;
            if(dataset.equals(datasetOpts[0]) && runLocal){
                conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1],
                                       flickr30k_mysqlParams[2], flickr30k_mysqlParams[3]);
            } else if(dataset.equals(datasetOpts[0])){
                conn = new DBConnector(flickr30k_sqlite);
            } else if(dataset.equals(datasetOpts[1]) && runLocal){
                conn = new DBConnector(mscoco_mysqlParams[0], mscoco_mysqlParams[1],
                                       mscoco_mysqlParams[2], mscoco_mysqlParams[3]);
            } else if(dataset.equals(datasetOpts[1])){
                conn = new DBConnector(mscoco_sqlite);
            }

            Integer numRandImgs = ns.getInt("rand_docs");
            Integer crossValFlag = null;
            switch(split){
                case "dev": crossValFlag = 0; break;
                case "train": crossValFlag = 1; break;
                case "test": crossValFlag = 2; break;
            }

            //get all documents
            if(split.equals("all")){
                docSet = DocumentLoader.getDocumentSet(conn);
            } else if(crossValFlag != null){
                if(numRandImgs == null)
                    docSet = DocumentLoader.getDocumentSet(conn, crossValFlag);
                else
                    docSet = DocumentLoader.getDocumentSet(conn, crossValFlag, numRandImgs);
            }
        }

        //Switch on the specified module, and parse module args
        List<String> argList = Arrays.asList(args);
        if(argList.contains("Debug")){
            if(ns.getBoolean("heterog_subset")) {
                Minion.export_subsetHeterogType(docSet);
            } else if(ns.getBoolean("penult_filter")){
                Minion.export_penultFilter(docSet);
            } else if(ns.getBoolean("mod_subset")){
                Minion.export_modSubsetFeats(docSet, split);
            } else {


                /*
                IllinoisTagger tggr_biff = new IllinoisTagger(Overlord.dataPath + "pos/");
                List<String> cocoLines_orig =
                        FileIO.readFile_lineList(Overlord.mscocoPath + "coco_caps.txt");
                List<String> cocoLines_new = new ArrayList<>();
                for(String line : cocoLines_orig){
                    String[] lineParts = line.split("\t");
                    edu.illinois.cs.cogcomp.lbjava.nlp.seg.Token[] toks = tggr_biff.predict(lineParts[1]);
                    StringBuilder sb = new StringBuilder();
                    sb.append(lineParts[0]);
                    sb.append("\t");
                    for(int i=0; i<toks.length; i++){
                        sb.append(toks[i].form);
                        if(i < toks.length-1)
                            sb.append(" ");
                    }
                    cocoLines_new.add(sb.toString());
                }
                FileIO.writeFile(cocoLines_new, "coco_caps", "token", false);
                System.exit(0);*/

                //Mention.initLexiconDict(Overlord.lexPath);
                /**/
                Caption.initLemmatizer();
                Cardinality.initCardLists(Overlord.flickr30kResources + "collectiveNouns.txt");
                String posDir = Overlord.dataPath + "pos/";
                String chunkDir = Overlord.dataPath + "chunk/";
                String cocoData = Overlord.mscocoPath + "coco_caps.txt";
                Map<String, Caption[]> captionDict = Minion.parseCocoCaptions(cocoData, posDir, chunkDir);

                Map<String, Caption> newCaps = new HashMap<>();
                for(Caption[] capArr : captionDict.values())
                    for(Caption c : capArr)
                        if(c != null)
                            newCaps.put(c.getUniqueID(), c);

                Map<String, Caption> xofyCaps = new HashMap<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        boolean isXofyCap = false;
                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            if(i < mentions.size() - 1){
                                List<Token> intrToks = c.getInterstitialTokens(mentions.get(i), mentions.get(i+1));
                                if(intrToks.size() == 1 && intrToks.get(0).toString().equals("of"))
                                    isXofyCap = true;
                            }
                        }
                        if(isXofyCap)
                            xofyCaps.put(c.getUniqueID(), c);
                    }
                }
                List<String> xofyCapIDs = new ArrayList<>(Minion.splitIDs);
                //List<String> xofyCapIDs = new ArrayList<>(xofyCaps.keySet());
                Collections.sort(xofyCapIDs);
                List<String> xofyCaps_diff = new ArrayList<>();
                for(String ID : xofyCapIDs) {
                    Caption c = xofyCaps.get(ID);
                    if(newCaps.containsKey(ID)){
                        String capStr_orig = c.toCorefString(false).trim();
                        String capStr_new = newCaps.get(ID).toCorefString(false).trim();
                        if(!capStr_orig.equals(capStr_new)){
                            xofyCaps_diff.add("orig: " + capStr_orig);
                            xofyCaps_diff.add("new : " + capStr_new);
                        }
                    }
                }
                FileIO.writeFile(xofyCaps_diff, "diff_cocoXofYCaps", "txt", true);
                System.exit(0);

                String[][] cocoLexTable =
                        FileIO.readFile_table(Overlord.mscocoResources + "coco_lex.csv");
                Map<String, Set<String>> catHeads = new HashMap<>();
                Map<String, Set<String>> catFallbacks = new HashMap<>();
                Map<String, String> superCats = new HashMap<>();
                for(String[] row : cocoLexTable){
                    String cat = row[0];
                    String[] heads = row[1].split("\\|");
                    String[] fallbacks = row[2].split("\\|");
                    String superCat = row[3];
                    catHeads.put(cat, new HashSet<>(Arrays.asList(heads)));
                    catFallbacks.put(cat, new HashSet<>(Arrays.asList(fallbacks)));
                    superCats.put(cat, superCat);
                }
                Mention.initLexiconDict(Overlord.lexPath);

                DoubleDict<String> hist = new DoubleDict<>();
                Map<String, List<Double>> mentionCounts = new HashMap<>();
                String[] mentionKeys = {"perCap", "coveredPerCap", "uncoveredPerCap",
                                        "scenePerCap", "coveredPerDoc", "uncoveredPerDoc",
                                        "scenePerDoc", "perDoc", "pronomPerCap",
                                        "pronomPerDoc"};
                for(String key : mentionKeys)
                    mentionCounts.put(key, new ArrayList<>());

                DoubleDict<String> uncoveredHeads = new DoubleDict<>();

                DoubleDict<String> imgs_head = new DoubleDict<>();
                DoubleDict<String> imgs_cat = new DoubleDict<>();
                DoubleDict<String> imgs_joint = new DoubleDict<>();
                for(Document d : docSet){
                    Set<String> headsPresent = new HashSet();
                    Set<String> catsPresent = new HashSet<>();
                    for(BoundingBox b : d.getBoundingBoxSet())
                        catsPresent.add(b.getCategory());

                    mentionCounts.get("perDoc").add((double)d.getMentionList().size());
                    int covered_perDoc = 0, unc_perDoc = 0, scene_perDoc = 0, pronom_perDoc = 0;

                    for(Caption c : d.getCaptionList()){
                        mentionCounts.get("perCap").add((double)c.getMentionList().size());
                        int covered_perCap = 0, unc_perCap = 0, scene_perCap = 0, pronom_perCap = 0;

                        for(Mention m : c.getMentionList()) {
                            String lastTwo = "";
                            List<Token> toks = m.getTokenList();
                            String head = m.getHead().getLemma().toLowerCase();

                            //TODO: Fix this temporary, hacky workaround for bad chunking
                            if (head.equals("standing"))
                                toks.remove(toks.size() - 1);
                            if (toks.isEmpty())
                                continue;

                            head = toks.get(toks.size()-1).getLemma().toLowerCase();
                            if (toks.size() > 1)
                                lastTwo = toks.get(toks.size() - 2).getLemma() + " ";
                            lastTwo += toks.get(toks.size() - 1).getLemma();
                            lastTwo = lastTwo.toLowerCase();

                            Set<String> trueCategories = new HashSet<>();
                            Set<String> fallbackCategories = new HashSet<>();
                            for (String cat : superCats.keySet()) {
                                if (catHeads.get(cat).contains(lastTwo) || catHeads.get(cat).contains(head)) {
                                    trueCategories.add(cat);
                                } else if (catFallbacks.get(cat).contains(lastTwo) || catFallbacks.get(cat).contains(head)) {
                                    fallbackCategories.add(cat);
                                }
                            }

                            hist.increment("mention_count");

                            String lexEntry = Mention.getLexicalEntry(head);
                            if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE){
                                hist.increment("pronom");
                                pronom_perCap++; pronom_perDoc++;
                            } else if (lexEntry != null && lexEntry.equals("scene")) {
                                hist.increment("scene");
                                scene_perCap++; scene_perDoc++;
                            } else if (trueCategories.isEmpty() && fallbackCategories.isEmpty()) {
                                hist.increment("uncovered");
                                unc_perCap++; unc_perDoc++;
                                uncoveredHeads.increment(head);
                                headsPresent.add(head);
                            } else {
                                hist.increment("covered");
                                covered_perCap++; covered_perDoc++;
                            }
                        }
                        mentionCounts.get("coveredPerCap").add((double)covered_perCap);
                        mentionCounts.get("uncoveredPerCap").add((double)unc_perCap);
                        mentionCounts.get("scenePerCap").add((double)scene_perCap);
                        mentionCounts.get("pronomPerCap").add((double)pronom_perCap);
                    }
                    mentionCounts.get("coveredPerDoc").add((double)covered_perDoc);
                    mentionCounts.get("uncoveredPerDoc").add((double)unc_perDoc);
                    mentionCounts.get("scenePerDoc").add((double)scene_perDoc);
                    mentionCounts.get("pronomPerDoc").add((double)pronom_perDoc);

                    for(String head : headsPresent)
                        imgs_head.increment(head);
                    for(String cat : catsPresent)
                        imgs_cat.increment(cat.replace(" ", "_").toUpperCase());
                    for(String head : headsPresent)
                        for(String cat : catsPresent)
                            imgs_joint.increment(cat.replace(" ", "_").toUpperCase() + "|" + head);
                }

                //remove everything lower than the cutoff
                int frequencyCuttoff = 5;
                Set<String> headsToRemove = new HashSet<>();
                for(String head : imgs_head.keySet())
                    if(imgs_head.get(head) < frequencyCuttoff)
                        headsToRemove.add(head);
                for(String head : headsToRemove)
                    imgs_head.remove(head);

                for(String key : imgs_head.keySet())
                    imgs_head.divide(key, docSet.size());
                for(String key : imgs_cat.keySet())
                    imgs_cat.divide(key, docSet.size());
                for(String key : imgs_joint.keySet())
                    imgs_joint.divide(key, docSet.size());

                OutTable ot_pmi = new OutTable("category", "head", "cat_prob", "head_prob", "joint_prob", "pmi");
                for(String cat : imgs_cat.keySet()){
                    double catProb = imgs_cat.get(cat);
                    for(String head : imgs_head.keySet()){
                        double headProb = imgs_head.get(head);
                        double jointProb = imgs_joint.get(cat + "|" + head);
                        if(jointProb > 0){
                            double pmi = StatisticalUtil.computePMI(catProb, headProb, jointProb, true);
                            ot_pmi.addRow(cat, head, catProb, headProb, jointProb, pmi);
                        }
                    }
                }
                ot_pmi.writeToCsv("hist_catHeadPmi", true);

                System.out.printf("Mention distribution - pronominal: %.2f%%; "+
                                  "scene: %.2f%%; covered: %.2f%%; uncovered: %.2f%%\n",
                                  100.0 * hist.get("pronom") / hist.get("mention_count"),
                                  100.0 * hist.get("scene") / hist.get("mention_count"),
                                  100.0 * hist.get("covered") / hist.get("mention_count"),
                                  100.0 * hist.get("uncovered") / hist.get("mention_count"));
                System.out.printf("Mentions per caption: %.2f; pronom: %.2f; scene: %.2f; covered: %.2f; uncovered: %.2f\n",
                        StatisticalUtil.getMean(mentionCounts.get("perCap")),
                        StatisticalUtil.getMean(mentionCounts.get("pronomPerCap")),
                        StatisticalUtil.getMean(mentionCounts.get("scenePerCap")),
                        StatisticalUtil.getMean(mentionCounts.get("coveredPerCap")),
                        StatisticalUtil.getMean(mentionCounts.get("uncoveredPerCap")));
                System.out.printf("Mentions per doc: %.2f; pronom: %.2f; scene: %.2f; covered: %.2f; uncovered: %.2f\n",
                        StatisticalUtil.getMean(mentionCounts.get("perDoc")),
                        StatisticalUtil.getMean(mentionCounts.get("pronomPerDoc")),
                        StatisticalUtil.getMean(mentionCounts.get("scenePerDoc")),
                        StatisticalUtil.getMean(mentionCounts.get("coveredPerDoc")),
                        StatisticalUtil.getMean(mentionCounts.get("uncoveredPerDoc")));

                Set<String> headsToDelete = new HashSet<>();
                for(String head : uncoveredHeads.keySet())
                    if(uncoveredHeads.get(head) < 6)
                        headsToDelete.add(head);
                for(String head : headsToDelete)
                    uncoveredHeads.remove(head);
                FileIO.writeFile(uncoveredHeads, "hist_uncoveredCocoHeads", "csv", true);

                //String cocoCorefFile = "/shared/projects/DenotationGraph/corpora/COCO_coref/COCO_coref.coref";
                //Minion.importCocoData(cocoCorefFile);
                System.exit(0);

                Minion.export_cocoCategoryStats_givenBox(docSet);
                Minion.export_cocoCategoryStats_givenMention(docSet);
                System.exit(0);

                String[] notableCats = {"handbag", "dining table", "potted plant",
                                        "backpack", "baseball glove", "spoon",
                                        "cup", "knife", "fork", "bottle", "oven"};
                /*
                for(String cat : notableCats){
                    List<String> ll_caps = new ArrayList<>();
                    for(Caption c : zeroMentionCaps.get(cat))
                        ll_caps.add(c.toString());
                    FileIO.writeFile(ll_caps, "out/charts/zeroMentionCaps_" + cat, "txt", true);
                }


                Map<String, DoubleDict<String>> cocoFreqDict = new HashMap<>();
                for(String cat : superCats.keySet())
                    cocoFreqDict.put(cat, new DoubleDict<>());


                for(Document d : docSet){
                    Set<String> categories = new HashSet<>();
                    d.getBoundingBoxSet().forEach(b -> categories.add(b.getCategory()));
                    for(Mention m : d.getMentionList()){
                        String head = m.getHead().getLemma().toLowerCase();
                        String lastTwo = "";
                        List<Token> toks = m.getTokenList();
                        if(toks.size() > 1)
                            lastTwo = toks.get(toks.size()-2).getLemma() + " ";
                        lastTwo += toks.get(toks.size()-1).getLemma();
                        lastTwo = lastTwo.toLowerCase();

                        //Increment the covered cases
                        boolean foundCat = false, foundFallback = false;
                        for(String cat : categories){
                            if(catHeads.get(cat).contains(lastTwo) || catHeads.get(cat).contains(head)){
                                cocoFreqDict.get(cat).increment(cat.toUpperCase());
                                foundCat = true;
                                //We could break here, but there shoudn't be overlap with this
                            }
                        }

                        //Increment the fallback cases
                        if(!foundCat){
                            for(String cat : categories){
                                if(catFallbacks.get(cat).contains(lastTwo) || catFallbacks.get(cat).contains(head)){
                                    cocoFreqDict.get(cat).increment("FALLBACK");
                                    foundFallback = true;
                                    //Dont break here, since we want to count each of the multiple fallbacks
                                }
                            }
                        }

                        //Increment the strings themselves if they're not even fallback options
                        if(!foundCat && !foundFallback){
                            for(String cat : categories){
                                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                                    cocoFreqDict.get(cat).increment("PRONOUN");
                                else
                                    cocoFreqDict.get(cat).increment(lastTwo);
                            }
                        }
                    }
                }

                OutTable ot_cocoFreq = new OutTable("category", "penultimate", "freq");
                for(String cat : cocoFreqDict.keySet()){
                    List<String> sortedPenults = cocoFreqDict.get(cat).getSortedByValKeys(true);
                    sortedPenults = sortedPenults.subList(0, Math.min(sortedPenults.size(), 100));
                    for(String penult : sortedPenults)
                        ot_cocoFreq.addRow(cat, penult, cocoFreqDict.get(cat).get(penult));
                }
                ot_cocoFreq.writeToCsv("hist_coco_cat_penults", true);
                System.exit(0);
*/

                //Minion.export_bryanPreproc_coco(docSet);
                //System.exit(0);
                Mention.initLexiconDict(lexPath);
                List<String> ll_snli = FileIO.readFile_lineList(Overlord.snliPath +
                        "snli_1.0_train.jsonl");
                IllinoisTagger tggr = new IllinoisTagger(Overlord.dataPath + "pos/");
                IllinoisChunker chnkr = new IllinoisChunker(Overlord.dataPath + "chunk/");
                Set<SNLIPair> snliPairs = new HashSet<>();
                for(String line : ll_snli)
                    snliPairs.add(new SNLIPair(line, tggr, chnkr));
                List<String> ll_premise = new ArrayList<>(), ll_hyp = new ArrayList<>();
                for(SNLIPair pair : snliPairs){
                    String pairID = pair.getID();
                    ll_premise.add(pairID + "\t" + pair.getPremise().toCorefString());
                    ll_hyp.add(pairID + "\t" + pair.getHypothesis().toCorefString());
                }
                FileIO.writeFile(ll_premise, Overlord.snliPath + "snli_train_premise.coref");
                FileIO.writeFile(ll_hyp, Overlord.snliPath + "snli_train_hypothesis.coref");
                System.exit(0);

                Map<String, String> cocoLexicon = Minion.createCocoLexicon(docSet);

                //For each mention, to how many boxes could it ground? (by type)

                Map<String, DoubleDict<Integer>> typeAmbiguityFreq_mention = new HashMap<>();
                Map<String, DoubleDict<Integer>> typeAmbiguityFreq_box = new HashMap<>();
                for(Document d : docSet){
                    DoubleDict<String> boxFreq = new DoubleDict<>();
                    for(BoundingBox b : d.getBoundingBoxSet())
                        boxFreq.increment(b.getCategory());
                    DoubleDict<String> mentionFreq = new DoubleDict<>();
                    for(Mention m : d.getMentionList()){
                        String lexEntry = cocoLexicon.get(m.toString().toLowerCase());
                        if(lexEntry != null)
                            for(String l : lexEntry.split("/"))
                                mentionFreq.increment(l);
                    }

                    for(Mention m : d.getMentionList()){
                        String lexEntry = cocoLexicon.get(m.toString().toLowerCase());
                        if(lexEntry != null){
                            int boxCount = 0;
                            String[] lexEntries = lexEntry.split("/");
                            for(String l : lexEntries)
                                if(!l.trim().isEmpty())
                                    boxCount += boxFreq.get(l.trim());
                            for(String l : lexEntries){
                                if(!typeAmbiguityFreq_mention.containsKey(l))
                                    typeAmbiguityFreq_mention.put(l, new DoubleDict<>());
                                typeAmbiguityFreq_mention.get(l).increment(boxCount);
                            }
                        }
                    }
                    for(BoundingBox b : d.getBoundingBoxSet()){
                        String cat = b.getCategory();
                        if(!typeAmbiguityFreq_box.containsKey(cat))
                            typeAmbiguityFreq_box.put(cat, new DoubleDict<>());
                        typeAmbiguityFreq_box.get(cat).increment((int)mentionFreq.get(cat));
                    }
                }
                OutTable ot_cocoTypeAmbiguity_mention = new OutTable("type", "possible_boxes", "freq", "perc");
                for(String type : typeAmbiguityFreq_mention.keySet()){
                    double total = typeAmbiguityFreq_mention.get(type).getSum();
                    for(Integer boxes : typeAmbiguityFreq_mention.get(type).keySet())
                        ot_cocoTypeAmbiguity_mention.addRow(type, boxes, typeAmbiguityFreq_mention.get(type).get(boxes),
                                typeAmbiguityFreq_mention.get(type).get(boxes) / total);
                }
                ot_cocoTypeAmbiguity_mention.writeToCsv("hist_cocoTypeAmbiguity_mention", true);
                OutTable ot_cocoTypeAmbiguity_box = new OutTable("type", "possible_mentions", "freq", "perc");
                for(String type : typeAmbiguityFreq_box.keySet()){
                    double total = typeAmbiguityFreq_box.get(type).getSum();
                    for(Integer mentions : typeAmbiguityFreq_box.get(type).keySet())
                        ot_cocoTypeAmbiguity_box.addRow(type, mentions, typeAmbiguityFreq_box.get(type).get(mentions),
                                typeAmbiguityFreq_box.get(type).get(mentions) / total);
                }
                ot_cocoTypeAmbiguity_box.writeToCsv("hist_cocoTypeAmbiguity_box", true);
                System.exit(0);


                DoubleDict<String> typeFreq_mb = new DoubleDict<>();    //Sum of average M/B values, per type
                DoubleDict<String> typeFreq_bm = new DoubleDict<>();    //Sum of average B/M values, per type
                DoubleDict<String> typeFreq_nobox = new DoubleDict<>(); //Sum of instances where type T appears in
                //a mention but not a box
                DoubleDict<String> typeFreq_nomen = new DoubleDict<>(); //Sum of instances where type T appears in
                //a box but not a mention
                for(Document d : docSet){
                    DoubleDict<String> typeFreq_mention = new DoubleDict<>();
                    DoubleDict<String> typeFreq_box = new DoubleDict<>();
                    for(Mention m : d.getMentionList()){
                        String mCat = cocoLexicon.get(m.toString().toLowerCase());
                        if(mCat != null && !mCat.contains("/"))
                            typeFreq_mention.increment(mCat);
                    }

                    for(BoundingBox b : d.getBoundingBoxSet())
                        typeFreq_box.increment(b.getCategory());

                    for(String type : typeFreq_mention.keySet()){
                        if(typeFreq_box.containsKey(type))
                            typeFreq_mb.increment(type, typeFreq_mention.get(type) / typeFreq_box.get(type));
                        else
                            typeFreq_nobox.increment(type);
                    }

                    for(String type : typeFreq_box.keySet()){
                        if(typeFreq_mention.containsKey(type))
                            typeFreq_bm.increment(type, typeFreq_box.get(type) / typeFreq_mention.get(type));
                        else
                            typeFreq_nomen.increment(type);
                    }
                }
                for(String type : typeFreq_mb.keySet())
                    typeFreq_mb.divide(type, docSet.size());
                for(String type : typeFreq_bm.keySet())
                    typeFreq_bm.divide(type, docSet.size());
                for(String type : typeFreq_nobox.keySet())
                    typeFreq_nobox.divide(type, docSet.size());
                for(String type : typeFreq_nomen.keySet())
                    typeFreq_nomen.divide(type, docSet.size());

                FileIO.writeFile(typeFreq_mb, "hist_coco_mentionsPerBox", "csv", true);
                FileIO.writeFile(typeFreq_bm, "hist_coco_boxesPerMention", "csv", true);
                FileIO.writeFile(typeFreq_nobox, "hist_coco_nobox", "csv", true);
                FileIO.writeFile(typeFreq_nomen, "hist_coco_nomention", "csv", true);

                System.exit(0);








                Logger.log("Initializing lexicons");
                Map<String, Set<String>> categoryLemmaDict = new HashMap<>();
                for(String[] row : FileIO.readFile_table(Overlord.mscocoResources + "coco_lex.csv")){
                    categoryLemmaDict.put(row[0], new HashSet<>());
                    for(int i=1; i<row.length; i++)
                        categoryLemmaDict.get(row[0]).add(row[i]);
                }
                Mention.initLexiconDict(Overlord.lexPath);
                Map<String, String> mTypeDict_coco = new HashMap<>();
                Map<String, String> mTypeDict_flickr = new HashMap<>();
                DoubleDict<String> typeFreq = new DoubleDict<>();
                DoubleDict<String> uncoveredLemmas = new DoubleDict<>();
                double totalMentions = 0.0;
                for(Document d : docSet){
                    totalMentions += d.getMentionList().size();
                    for(Mention m : d.getMentionList()){
                        //Determine if this mention has a coco category (or more than one)
                        String mStr = m.toString().toLowerCase();
                        List<Token> toks = m.getTokenList();
                        String lastLem = toks.get(toks.size()-1).getLemma().toLowerCase();

                        //NOTE: This is a temporary, hacky workaround for bad chunking; ADDRESS THIS
                        if(lastLem.equals("standing"))
                            toks.remove(toks.size()-1);
                        if(toks.isEmpty())
                            continue;

                        lastLem = toks.get(toks.size()-1).getLemma().toLowerCase();
                        String lastTwoLems = "";
                        if(toks.size() > 1)
                            lastTwoLems = toks.get(toks.size()-2).getLemma().toLowerCase() + " " + lastLem;

                        String mType = null;
                        boolean cocoType = true;
                        if(mTypeDict_coco.containsKey(mStr)){
                            mType = mTypeDict_coco.get(mStr);
                        } else if (mTypeDict_flickr.containsKey(mStr)){
                            mType = mTypeDict_flickr.get(mStr);
                            cocoType = false;
                        } else {
                            //1) Determine if this mention has an MSCOCO type
                            if(!lastLem.isEmpty()){
                                Set<String> types = new HashSet<>();
                                for(String cat : categoryLemmaDict.keySet())
                                    for(String lemma : categoryLemmaDict.get(cat))
                                        if(lemma.equals(lastLem) || lemma.equals(lastTwoLems))
                                            types.add(cat);
                                if(!types.isEmpty()){
                                    List<String> typeList = new ArrayList<>(types);
                                    Collections.sort(typeList);
                                    mType = StringUtil.listToString(typeList, "/");
                                }
                            }
                            //2) If we didn't find a coco type, check to see if
                            //   this is a flickr type
                            if(mType == null && !lastLem.isEmpty()){
                                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                                    mType = m.getPronounType().toString().toUpperCase();
                                else{
                                    String flickrType = Mention.getLexicalEntry(lastLem);
                                    if(flickrType != null && !flickrType.equals("other"))
                                        mType = flickrType.toUpperCase();
                                }

                                if(mType != null)
                                    cocoType = false;
                            }
                        }

                        //If we haven't found a type, this is an unknown mention
                        if(mType == null){
                            uncoveredLemmas.increment(lastLem);
                        } else {
                            //if we have, record it
                            if(cocoType && !mTypeDict_coco.containsKey(mType))
                                mTypeDict_coco.put(mStr, mType);
                            else if(!cocoType && !mTypeDict_flickr.containsKey(mType))
                                mTypeDict_flickr.put(mStr, mType);

                            typeFreq.increment(mType);
                        }
                    }
                }

                Logger.log("Lexicon coverage");
                Set<String> cocoTypes = new HashSet<>(mTypeDict_coco.values());
                Set<String> flickrTypes = new HashSet<>(mTypeDict_flickr.values());
                double cocoMentions = 0.0, flickrMentions = 0.0, cocoMentions_mult = 0.0;
                for(String type : typeFreq.keySet()){
                    if(cocoTypes.contains(type))
                        cocoMentions += typeFreq.get(type);
                    else if(flickrTypes.contains(type))
                        flickrMentions += typeFreq.get(type);

                    if(cocoTypes.contains(type) && type.contains("/"))
                        cocoMentions_mult += typeFreq.get(type);
                }
                System.out.printf("COCO: %.2f%%; Flickr: %.2f%%; Neither: %.2f%%\n",
                                  100.0 * cocoMentions / totalMentions,
                                  100.0 * flickrMentions / totalMentions,
                                  100.0 * (totalMentions - cocoMentions - flickrMentions) / totalMentions);
                System.out.printf("mult_coco: %.2f%%\n", 100.0 * cocoMentions_mult / totalMentions);
                Logger.log("Writing full histogram to file");
                FileIO.writeFile(typeFreq, "hist_cocoTypes", "csv", true);

                Logger.log("Writing uncovered lemmas to file");
                FileIO.writeFile(uncoveredLemmas, "hist_uncovered_cocoLems", "csv", true);
                System.exit(0);



                Map<String, String> xTypeDict = new HashMap<>();
                String[][] xofyTable =
                        FileIO.readFile_table(Overlord.dataPath + "hist_xofy_lemma_coco_20170424.csv");
                for(int i=1; i<xofyTable.length; i++){
                    String[] row = xofyTable[i];
                    if(!row[0].isEmpty() && !row[1].isEmpty())
                        xTypeDict.put(row[0].replace("\"", ""), row[1].replace("\"", ""));
                }
                Set<String> xTypes = new HashSet<>();
                for(String xLemma : xTypeDict.keySet())
                    xTypes.add(xTypeDict.get(xLemma));
                for(String xType : xTypes)
                    System.out.println(xType);
                System.exit(0);

                Map<String, DoubleDict<String>> xofYLemmas = new HashMap<>();

                DoubleDict<String> xLemmaHist = new DoubleDict<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            Mention m_j = null;
                            if(i < mentions.size() - 1)
                                m_j = mentions.get(i+1);

                            String xLemma = null, yLemma = null;
                            if(m_i.toString().toLowerCase().contains(" of ")){
                                List<Token> xToks = new ArrayList<>();
                                for(Token t : m_i.getTokenList()){
                                    if(t.toString().toLowerCase().equals("of"))
                                        break;
                                    else
                                        xToks.add(t);
                                }
                                xLemma = xToks.get(xToks.size() - 1).getLemma().toLowerCase();
                                yLemma = m_i.getHead().getLemma().toLowerCase();
                            } else if(m_j != null){
                                List<Token> intrsttlToks = c.getInterstitialTokens(m_i, m_j);
                                if(intrsttlToks.size() == 1 && intrsttlToks.get(0).toString().toLowerCase().equals("of")){
                                    xLemma = m_i.getHead().getLemma().toLowerCase();
                                    yLemma = m_j.getHead().getLemma().toLowerCase();
                                }
                            }

                            //if this is an XofY construction (or X in XofY), log it
                            if(xLemma != null){
                                xLemmaHist.increment(xLemma);
                                if(!xofYLemmas.containsKey(xLemma))
                                    xofYLemmas.put(xLemma, new DoubleDict<>());
                                xofYLemmas.get(xLemma).increment(yLemma);
                            }
                        }
                    }
                }


                double totalXofY = xLemmaHist.getSum();
                OutTable ot_xofy_coco = new OutTable("x_lemma", "category", "freq", "perc", "top_y_heads");
                for(String xLemma : xLemmaHist.getSortedByValKeys(true)){
                    double xFreq = xLemmaHist.get(xLemma);
                    String xType = "unk";
                    if(xTypeDict.containsKey(xLemma))
                        xType = xTypeDict.get(xLemma);
                    List<String> yLemmaFreqs = new ArrayList<>();
                    for(String yLemma : xofYLemmas.get(xLemma).getSortedByValKeys(true))
                        yLemmaFreqs.add(String.format("%s (%.2f%%)", yLemma,
                                xofYLemmas.get(xLemma).get(yLemma) / xFreq));
                    ot_xofy_coco.addRow(xLemma, xType, (int)xFreq, xFreq / totalXofY,
                            StringUtil.listToString(yLemmaFreqs, " | "));
                }
                ot_xofy_coco.writeToCsv("hist_xofy_lemmas_coco", true);
                System.exit(0);



                /*
                container
                quant-whole
                unk
                portion
                part-of
                collective
                relation-near
                quantifier
                nonvisual
                representation
                 */



                /*
                String[] cocoCatArr = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
                Set<String> cocoCatSet = new HashSet<>();
                cocoCatSet.addAll(Arrays.asList(cocoCatArr));*/

                Logger.log("Loading wordnet");
                WordnetUtil wnUtil = new WordnetUtil(wordnetDir);

                Logger.log("Loading coco lexicon");
                Map<String, Set<String>> catLemmaDict = new HashMap<>();
                String[][] catLemmaTable = FileIO.readFile_table(Overlord.mscocoPath + "coco_lex.csv");
                for(String[] row : catLemmaTable){
                    catLemmaDict.put(row[0], new HashSet<>());
                    catLemmaDict.get(row[0]).add(row[0]); //Each category contains its own label
                    for(int i=1; i<row.length; i++)
                        if(!row[i].trim().isEmpty())
                            catLemmaDict.get(row[0]).add(row[i]);
                }

                Logger.log("Parsing coco types");
                DoubleDict<String> knownMentions = new DoubleDict<>();
                DoubleDict<String> unkMentions = new DoubleDict<>();
                Map<String, String> knownMentionDict = new HashMap<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        String mStr = m.toString().toLowerCase();
                        List<Token> toks = m.getTokenList();
                        String lastLem = toks.get(toks.size()-1).getLemma().toLowerCase();

                        //NOTE: This is a temporary, hacky workaround for bad chunking; ADDRESS THIS
                        if(lastLem.equals("standing"))
                            toks.remove(toks.size()-1);
                        if(toks.isEmpty())
                            continue;

                        lastLem = toks.get(toks.size()-1).getLemma().toLowerCase();
                        String lastTwoLems = "";
                        if(toks.size() > 1)
                            lastTwoLems = toks.get(toks.size()-2).getLemma().toLowerCase() + " " + lastLem;

                        if(knownMentions.containsKey(mStr)) {
                            knownMentions.increment(mStr);
                        } else if(unkMentions.containsKey(mStr)) {
                            unkMentions.increment(mStr);
                        } else {
                            //If we haven't encountered this mention before, determine
                            //if this has a known or unknown type
                            Set<String> categories = new HashSet<>();

                            if(!lastLem.isEmpty()){
                                //1) Check if this mention ends in one (or more) of the
                                //   coco lexicon entries, we know its type
                                for(String cat : catLemmaDict.keySet())
                                    for(String lemma : catLemmaDict.get(cat))
                                        if(lemma.equals(lastLem) || lemma.equals(lastTwoLems))
                                            categories.add(cat);

                                //2) If we found no categories in the lexicon, check wordnet
                                if(categories.isEmpty()){
                                    HypTree tree = wnUtil.getHypernymTree(lastLem);
                                    for(List<HypTree.HypNode> branch : tree.getRootBranches()){
                                        for(HypTree.HypNode leaf : branch){
                                            String hyp = leaf.toString();
                                            if(catLemmaDict.containsKey(hyp))
                                                categories.add(hyp);
                                        }
                                    }
                                }
                            }

                            //3) Finally, add this mention string to the appropriate
                            //   group, adding its lemma to the category dict if its
                            //   not already there
                            if(categories.isEmpty()){
                                unkMentions.increment(mStr);
                            } else {
                                String catStr = StringUtil.listToString(categories, "|");
                                knownMentions.increment(mStr);
                                knownMentionDict.put(mStr, catStr);
                                for(String cat : categories)
                                    catLemmaDict.get(cat).add(lastLem);
                            }
                        }
                    }
                }

                System.out.printf("Found coco categories for %.2f%% of mentions\n",
                        100.0 * knownMentions.getSum() /
                        (knownMentions.getSum() + unkMentions.getSum()));

                Logger.log("Writing files");
                OutTable ot_knwn = new OutTable("mention", "freq", "coco_category");
                OutTable ot_unk = new OutTable("mention", "freq");
                for(String mention : knownMentions.getSortedByValKeys())
                    ot_knwn.addRow(mention, knownMentions.get(mention), knownMentionDict.get(mention));
                for(String mention : unkMentions.getSortedByValKeys())
                    ot_unk.addRow(mention, unkMentions.get(mention));
                ot_knwn.writeToCsv("hist_coco_knownMentions", true);
                ot_unk.writeToCsv("hist_coco_unkMentions", true);

                List<String> ll_cocoCats = new ArrayList<>();
                for(String cat : catLemmaDict.keySet()){
                    StringBuilder line = new StringBuilder();
                    line.append(cat);
                    for(String lemma : catLemmaDict.get(cat)){
                        line.append(",");
                        line.append(lemma);
                    }
                    ll_cocoCats.add(line.toString());
                }
                FileIO.writeFile(ll_cocoCats, "coco_lex", "csv", false);
                System.exit(0);



                /*
                List<Double> boxCounts = new ArrayList<>();
                for(Document d : docSet)
                    boxCounts.add((double)d.getBoundingBoxSet().size());
                System.out.println(StatisticalUtil.getMean(boxCounts));
                */
                //DBConnector conn = new DBConnector("COCO_" + Util.getCurrentDateTime("yyyyMMdd") + ".db");
               //DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1],
               //        flickr30k_mysqlParams[2], "ccervan2_coco");

                /*
                Collection<Document> docSet_coco =
                        DocumentLoader.getDocumentSet(Overlord.dataPath + "mscoco/coco_train_sub_20170331.coref",
                                Overlord.lexPath, Overlord.flickr30kResources);
                double mentionCount = 0.0;
                for(Document d : docSet_coco)
                    mentionCount += d.getMentionList().size();
                System.out.println(docSet_coco.size());
                System.out.println(mentionCount / (5*docSet_coco.size()));
                System.exit(0);*/


                DoubleDict<String> hist_heur = new DoubleDict<>(), hist_intr = new DoubleDict<>();
                Map<Mention, Set<Mention>> heur_cloth = new HashMap<>(), heur_part = new HashMap<>(),
                        intr_cloth = new HashMap<>(), intr_part = new HashMap<>();
                DoubleDict<String> hist_interstitial = new DoubleDict<>();

                //get the heuristic-attached cloth and bodyparts
                Map<Mention, AttrStruct> attrDict =
                        ClassifyUtil.attributeAttachment_agent(docSet);
                for(Mention m_agent : attrDict.keySet()){
                    AttrStruct as = attrDict.get(m_agent);
                    Set<Mention> clothing = new HashSet<>(), bodyparts = new HashSet<>();
                    for(Mention m_attr : as.getAttributeMentions()){
                        String lexType = m_attr.getLexicalType();
                        if(lexType.contains("bodyparts"))
                            bodyparts.add(m_attr);
                        else if(lexType.contains("clothing") || lexType.contains("colors"))
                            clothing.add(m_attr);
                    }
                    if(!clothing.isEmpty()){
                        if(!heur_cloth.containsKey(m_agent))
                            heur_cloth.put(m_agent, new HashSet<>());
                        heur_cloth.get(m_agent).addAll(clothing);
                    }
                    if(!bodyparts.isEmpty()){
                        if(!heur_part.containsKey(m_agent))
                            heur_part.put(m_agent, new HashSet<>());
                        heur_part.get(m_agent).addAll(bodyparts);
                    }

                    for(Mention m : clothing){
                        String pair_code;
                        if(m_agent.getIdx() < m.getIdx())
                            pair_code = "agent|cloth";
                        else
                            pair_code = "cloth|agent";
                        hist_heur.increment(pair_code);
                    }
                    for(Mention m : bodyparts){
                        String pair_code;
                        if(m_agent.getIdx() < m.getIdx())
                            pair_code = "agent|part";
                        else
                            pair_code = "part|agent";
                        hist_heur.increment(pair_code);
                    }
                }

                //Get the agent/bodyparts and agent/clothing that are separated by
                //verbs (subj/obj or interstitial) or preps (interstitial)
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        //get subject/object pairings
                        Map<Mention, Chunk> mentionVerbDict_subj = new HashMap<>();
                        Map<Mention, Chunk> mentionVerbDict_obj = new HashMap<>();
                        for(Mention m : c.getMentionList()){
                            Chunk verb_subj = c.getSubjectOf(m);
                            Chunk verb_obj = c.getObjectOf(m);
                            if(verb_subj != null)
                                mentionVerbDict_subj.put(m, verb_subj);
                            if(verb_obj != null)
                                mentionVerbDict_obj.put(m, verb_obj);
                        }
                        Map<Chunk, Set<Mention>> verbMentionDict_subj =
                                Util.invertMap(mentionVerbDict_subj);
                        Map<Chunk, Set<Mention>> verbMentionDict_obj =
                                Util.invertMap(mentionVerbDict_obj);
                        for(Chunk verb_subj : verbMentionDict_subj.keySet()){
                            Set<Mention> subjOfMentions = verbMentionDict_subj.get(verb_subj);
                            if(verbMentionDict_obj.containsKey(verb_subj)){
                                Set<Mention> objOfMentions = verbMentionDict_obj.get(verb_subj);
                                for(Mention m_subj : subjOfMentions){
                                    String lexType_subj = m_subj.getLexicalType();
                                    for(Mention m_obj : objOfMentions) {
                                        String lexType_obj = m_obj.getLexicalType();

                                        boolean agent_subj = false, agent_obj=false,
                                                part_subj = false, part_obj = false,
                                                cloth_subj = false, cloth_obj = false;
                                        if(m_subj.getPronounType().isAnimate() || lexType_subj.equals("people") ||
                                           lexType_subj.equals("animals"))
                                            agent_subj = true;
                                        if(m_obj.getPronounType().isAnimate() || lexType_obj.equals("people") ||
                                                lexType_obj.equals("animals"))
                                            agent_obj = true;
                                        if(lexType_subj.equals("bodyparts"))
                                            part_subj = true;
                                        if(lexType_obj.equals("bodyparts"))
                                            part_obj = true;
                                        if(lexType_subj.equals("clothing") || lexType_subj.equals("colors") || lexType_subj.equals("clothing/colors"))
                                            cloth_subj = true;
                                        if(lexType_obj.equals("clothing") || lexType_obj.equals("colors") || lexType_obj.equals("clothing/colors"))
                                            cloth_obj = true;

                                        //determine what kind of pairing this is
                                        String pair_code = null;
                                        if(agent_subj){
                                            if(part_obj){
                                                if(!intr_part.containsKey(m_subj))
                                                    intr_part.put(m_subj, new HashSet<>());
                                                intr_part.get(m_subj).add(m_obj);
                                                if(m_subj.getIdx() < m_obj.getIdx())
                                                    pair_code = "agent|part";
                                                else
                                                    pair_code = "part|agent";
                                            } else if(cloth_obj){
                                                if(!intr_cloth.containsKey(m_subj))
                                                    intr_cloth.put(m_subj, new HashSet<>());
                                                intr_cloth.get(m_subj).add(m_obj);
                                                if(m_subj.getIdx() < m_obj.getIdx())
                                                    pair_code = "agent|cloth";
                                                else
                                                    pair_code = "cloth|agent";
                                            }
                                        } else if (agent_obj) {
                                            if(part_subj){
                                                if(!intr_part.containsKey(m_obj))
                                                    intr_part.put(m_obj, new HashSet<>());
                                                intr_part.get(m_obj).add(m_subj);
                                                if(m_obj.getIdx() < m_subj.getIdx())
                                                    pair_code = "agent|part";
                                                else
                                                    pair_code = "part|agent";
                                            } else if(cloth_subj){
                                                if(!intr_cloth.containsKey(m_obj))
                                                    intr_cloth.put(m_obj, new HashSet<>());
                                                intr_cloth.get(m_obj).add(m_subj);
                                                if(m_obj.getIdx() < m_subj.getIdx())
                                                    pair_code = "agent|cloth";
                                                else
                                                    pair_code = "cloth|agent";
                                            }
                                        }

                                        if(pair_code != null) {
                                            hist_intr.increment(pair_code);
                                            hist_interstitial.increment(verb_subj.getChunkType());
                                            hist_interstitial.increment(verb_subj.toString().toLowerCase());
                                        }
                                    }
                                }
                            }
                        }

                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            String lexType_i = m_i.getLexicalType();
                            for(int j=i+1; j<mentions.size(); j++){
                                Mention m_j = mentions.get(j);
                                String lexType_j = m_j.getLexicalType();
                                List<Chunk> intrsttl = c.getInterstitialChunks(m_i, m_j);
                                if(intrsttl.size() == 1){
                                    String pair_case = "";
                                    if(lexType_i.equals("people") || lexType_i.equals("animals") || m_i.getPronounType().isAnimate()){
                                        if(lexType_j.equals("bodyparts"))
                                            pair_case = "agent|part";
                                        else if(lexType_j.equals("clothing") || lexType_j.equals("colors") || lexType_j.equals("clothing/colors"))
                                            pair_case = "agent|cloth";
                                    } else if(lexType_j.equals("people") || lexType_j.equals("animals") || m_j.getPronounType().isAnimate()){
                                        if(lexType_i.equals("bodyparts"))
                                            pair_case = "part|agent";
                                        else if(lexType_i.equals("clothing") || lexType_i.equals("colors") || lexType_i.equals("clothing/colors"))
                                            pair_case = "cloth|agent";
                                    }

                                    if(pair_case.equals("agent|part")){
                                        if(!intr_part.containsKey(m_i))
                                            intr_part.put(m_i, new HashSet<>());
                                        if(!intr_part.get(m_i).contains(m_j)){
                                            intr_part.get(m_i).add(m_j);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("part|agent")){
                                        if(!intr_part.containsKey(m_j))
                                            intr_part.put(m_j, new HashSet<>());
                                        if(!intr_part.get(m_j).contains(m_i)){
                                            intr_part.get(m_j).add(m_i);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("agent|cloth")){
                                        if(!intr_cloth.containsKey(m_i))
                                            intr_cloth.put(m_i, new HashSet<>());
                                        if(!intr_cloth.get(m_i).contains(m_j)){
                                            intr_cloth.get(m_i).add(m_j);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    } else if(pair_case.equals("cloth|agent")){
                                        if(!intr_cloth.containsKey(m_j))
                                            intr_cloth.put(m_j, new HashSet<>());
                                        if(!intr_cloth.get(m_j).contains(m_i)){
                                            intr_cloth.get(m_j).add(m_i);
                                            hist_intr.increment(pair_case);

                                            hist_interstitial.increment(intrsttl.get(0).toString().toLowerCase());
                                            hist_interstitial.increment(intrsttl.get(0).getChunkType());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Logger.log("Heuristic Attachment");
                System.out.println(hist_heur);
                Logger.log("Verb/Interstitial Attachment");
                System.out.println(hist_intr);
                FileIO.writeFile(hist_interstitial, "hist_interstitial_attatch", "csv", true);


                Set<String> pairIDs_parts_heur = new HashSet<>(), pairIDs_cloth_heur = new HashSet<>(),
                        pairIDs_parts_intr = new HashSet<>(), pairIDs_cloth_intr = new HashSet<>();
                for(Mention agent : heur_cloth.keySet())
                    for(Mention cloth : heur_cloth.get(agent))
                        pairIDs_cloth_heur.add(Document.getMentionPairStr(agent, cloth));
                for(Mention agent : heur_part.keySet())
                    for(Mention part : heur_part.get(agent))
                        pairIDs_parts_heur.add(Document.getMentionPairStr(agent, part));
                for(Mention agent : intr_cloth.keySet())
                    for(Mention cloth : intr_cloth.get(agent))
                        pairIDs_cloth_intr.add(Document.getMentionPairStr(agent, cloth));
                for(Mention agent : intr_part.keySet())
                    for(Mention part : intr_part.get(agent))
                        pairIDs_parts_intr.add(Document.getMentionPairStr(agent, part));

                Logger.log("Heuristic Pairs");
                System.out.printf("cloth: %d; parts: %d\n", pairIDs_cloth_heur.size(), pairIDs_parts_heur.size());
                Logger.log("Interstitial Pairs");
                System.out.printf("cloth: %d; parts: %d\n", pairIDs_cloth_intr.size(), pairIDs_parts_intr.size());
                Logger.log("Intersection");
                Set<String> intersect_cloth = new HashSet<>(pairIDs_cloth_heur);
                intersect_cloth.retainAll(pairIDs_cloth_intr);
                Set<String> intersect_parts = new HashSet<>(pairIDs_parts_heur);
                intersect_parts.retainAll(pairIDs_parts_intr);
                System.out.printf("cloth: %d; parts: %d\n", intersect_cloth.size(), intersect_parts.size());



                System.exit(0);





                DoubleDict<String> hist_relation = new DoubleDict<>();
                DoubleDict<String> verbHist_so = new DoubleDict<>();
                DoubleDict<String> verbHist_intrsttl = new DoubleDict<>();
                DoubleDict<String> prepHist = new DoubleDict<>();
                for(Document d : docSet){


                    for(Caption c : d.getCaptionList()){
                        int numMentions = c.getMentionList().size();
                        hist_relation.increment("total_pairs",
                                0.5 * numMentions * (numMentions - 1));

                        Set<Mention[]> mentionPairs_verb = new HashSet<>();
                        Set<Mention[]> mentionPairs_prep = new HashSet<>();
                        Set<Mention[]> mentionPairs_intrstlVerb = new HashSet<>();

                        Map<Mention, Chunk> mentionVerbDict_subj = new HashMap<>();
                        Map<Mention, Chunk> mentionVerbDict_obj = new HashMap<>();
                        for(Mention m : c.getMentionList()){
                            Chunk verb_subj = c.getSubjectOf(m);
                            Chunk verb_obj = c.getObjectOf(m);
                            if(verb_subj != null)
                                mentionVerbDict_subj.put(m, verb_subj);
                            if(verb_obj != null)
                                mentionVerbDict_obj.put(m, verb_obj);
                        }
                        Map<Chunk, Set<Mention>> verbMentionDict_subj =
                                Util.invertMap(mentionVerbDict_subj);
                        Map<Chunk, Set<Mention>> verbMentionDict_obj =
                                Util.invertMap(mentionVerbDict_obj);
                        for(Chunk verb_subj : verbMentionDict_subj.keySet()){
                            Set<Mention> subjOfMentions = verbMentionDict_subj.get(verb_subj);
                            if(verbMentionDict_obj.containsKey(verb_subj)){
                                Set<Mention> objOfMentions = verbMentionDict_obj.get(verb_subj);
                                for(Mention m_subj : subjOfMentions){
                                    for(Mention m_obj : objOfMentions) {
                                        Mention[] pair = {m_subj, m_obj};
                                        if(!Util.containsArr(mentionPairs_verb, pair)) {
                                            mentionPairs_verb.add(pair);
                                            List<Token> toks = verb_subj.getTokenList();
                                            verbHist_so.increment("\""+toks.get(toks.size()-1).getLemma().toLowerCase()+"\"");
                                        }
                                    }
                                }
                            }
                        }

                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            for(int j=i+1; j<mentions.size(); j++){
                                Mention m_j = mentions.get(j);
                                List<Chunk> intrsttl = c.getInterstitialChunks(m_i, m_j);
                                if(intrsttl.size() == 1){
                                    Mention[] pair = {m_i, m_j};
                                    if(intrsttl.get(0).getChunkType().equals("VP")){
                                        if(!Util.containsArr(mentionPairs_intrstlVerb, pair)) {
                                            mentionPairs_intrstlVerb.add(pair);
                                            List<Token> toks = intrsttl.get(0).getTokenList();
                                            verbHist_intrsttl.increment("\""+toks.get(toks.size()-1).getLemma().toLowerCase()+"\"");
                                        }
                                    } else if(intrsttl.get(0).getChunkType().equals("PP")){
                                        if(!Util.containsArr(mentionPairs_prep, pair)) {
                                            mentionPairs_prep.add(pair);
                                            prepHist.increment("\""+intrsttl.get(0).toString().toLowerCase()+"\"");
                                        }
                                    }
                                }
                            }
                        }

                        hist_relation.increment("verb_so_pairs", mentionPairs_verb.size());
                        hist_relation.increment("prep_pairs", mentionPairs_prep.size());
                        hist_relation.increment("verb_pairs", mentionPairs_intrstlVerb.size());

                        int verbIntersect = 0, prepIntersect = 0;
                        for(Mention[] pair_verb : mentionPairs_verb){
                            if(Util.containsArr(mentionPairs_intrstlVerb, pair_verb))
                                verbIntersect++;
                            if(Util.containsArr(mentionPairs_prep, pair_verb))
                                prepIntersect++;
                        }
                        hist_relation.increment("verb_intersect", verbIntersect);
                        hist_relation.increment("prep_intersect", prepIntersect);
                    }
                }
                for(String rel : hist_relation.getSortedByValKeys(true))
                    System.out.printf("%s : %d (%.2f%%)\n", rel, (int)hist_relation.get(rel),
                            100.0 * hist_relation.get(rel) / hist_relation.get("total_pairs"));
                FileIO.writeFile(verbHist_so, "hist_verb_subjObj", "csv", true);
                FileIO.writeFile(verbHist_intrsttl, "hist_verb_intrsttl", "csv", true);
                FileIO.writeFile(prepHist, "hist_prep", "csv", true);
                System.exit(0);

                DoubleDict<String> hist_heads = new DoubleDict<>();
                Map<String, DoubleDict<String>> yHist_perX = new HashMap<>();
                DoubleDict<String> hist_chunkTypes = new DoubleDict<>();
                DoubleDict<String> hist_chunkStrs = new DoubleDict<>();
                DoubleDict<String> hist_xblanky = new DoubleDict<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> mentions = c.getMentionList();
                        for(int i=0; i<mentions.size(); i++){
                            Mention m_i = mentions.get(i);
                            String xHead = m_i.getHead().getLemma().toLowerCase();
                            if(i < mentions.size() - 1){
                                Mention m_j = mentions.get(i+1);
                                String yHead = m_j.getHead().getLemma().toLowerCase();

                                List<Chunk> interstChunks = c.getInterstitialChunks(m_i, m_j);

                                //List<Token> interstToks = c.getInterstitialTokens(m_i, m_j);
                                //if(interstToks.size() == 1 && interstToks.get(0).toString().equals("of")){

                                if(interstChunks.size() == 1){
                                    hist_chunkTypes.increment(interstChunks.get(0).getChunkType());
                                    hist_chunkStrs.increment(interstChunks.get(0).toString());

                                    hist_heads.increment(xHead);
                                    if(!yHist_perX.containsKey(xHead))
                                        yHist_perX.put(xHead, new DoubleDict<>());
                                    yHist_perX.get(xHead).increment(yHead);

                                    hist_xblanky.increment("\"" + xHead + "\",\"" +
                                            interstChunks.get(0).toString().toLowerCase() +
                                            "\",\"" + yHead + "\"");
                                }
                            }
                        }
                    }
                }
                List<String> ll_hist = new ArrayList<>();
                double ttl = hist_chunkStrs.getSum();
                for(String chunkStr : hist_chunkStrs.getSortedByValKeys(true))
                    ll_hist.add(String.format("\"%s\",%d,%f", chunkStr, (int)hist_chunkStrs.get(chunkStr),
                            hist_chunkStrs.get(chunkStr) / ttl));
                FileIO.writeFile(ll_hist, "hist_interstitial_chunks", "csv", true);
                ttl = hist_chunkTypes.getSum();
                for(String chunkType : hist_chunkTypes.getSortedByValKeys(true))
                    System.out.printf("%s: %d (%.2f%%)\n", chunkType, (int)hist_chunkTypes.get(chunkType),
                            100.0 * hist_chunkTypes.get(chunkType) / ttl);

                List<String> ll_xblanky = new ArrayList<>();
                ttl = hist_xblanky.getSum();
                for(String xblanky : hist_xblanky.getSortedByValKeys(true))
                    ll_xblanky.add(xblanky + "," + hist_xblanky.get(xblanky) +
                            "," + (hist_xblanky.get(xblanky)/ ttl));
                FileIO.writeFile(ll_xblanky, "ex_xblanky", "csv", true);
                System.exit(0);

                OutTable ot_xofy = new OutTable("x_head", "category", "freq", "y_head");
                double ttl_xofy = hist_heads.getSum();
                for(String xHead : hist_heads.getSortedByValKeys(true)){
                    List<String> yHeads =
                            new ArrayList<>(yHist_perX.get(xHead).getSortedByValKeys(true));
                    double totalYCount = yHist_perX.get(xHead).getSum();
                    List<String> yHeadWithFreqs = new ArrayList<>();
                    for(int i=0; i<Math.min(10, yHeads.size()); i++){
                        String yHead = yHeads.get(i);
                        yHeadWithFreqs.add(String.format("%s (%.2f%%) ",
                                yHead, 100.0 * yHist_perX.get(xHead).get(yHead) / totalYCount));
                    }
                    ot_xofy.addRow(xHead, "", (int)hist_heads.get(xHead),
                            hist_heads.get(xHead) / ttl_xofy,
                            StringUtil.listToString(yHeadWithFreqs, "|"));
                }
                ot_xofy.writeToCsv("hist_xofy_lemmas", true);

                System.exit(0);
                /*
                Set<String> imgIDs = new HashSet<>(); imgIDs.add("76739724.jpg");
                DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2],flickr30k_mysqlParams[3]);
                docSet = DocumentLoader.getDocumentSet(conn, imgIDs);*/


                Map<Mention, AttrStruct> attributeDict =
                        ClassifyUtil.attributeAttachment_agent(docSet);
                DoubleDict<AttrStruct> attrFreqs = new DoubleDict<>();
                Map<Mention, String> mentionCapDict = new HashMap<>();
                for(Document d : docSet)
                    for(Caption c : d.getCaptionList())
                        for(Mention m : c.getMentionList())
                            mentionCapDict.put(m, c.toString());

                for(Mention m : attributeDict.keySet()){
                    AttrStruct as = attributeDict.get(m);
                    attrFreqs.increment(as, as.getNumAttributes());
                }
                int attr_idx = 0;
                List<String> ll_attr = new ArrayList<>();
                for(AttrStruct as : attrFreqs.getSortedByValKeys(true)){
                    attr_idx++;
                    if(attr_idx < 10) {
                        ll_attr.add(as.toLatexString());
                        String cap = "";
                        for(Mention m : as.getAttributeMentions())
                            if(mentionCapDict.containsKey(m))
                                cap = mentionCapDict.get(m);
                        ll_attr.add(cap);
                        ll_attr.add("");
                    }
                }
                attr_idx = 0;
                for(AttrStruct as : attrFreqs.getSortedByValKeys(false)){
                    if(!as.getAttributeMentions().isEmpty()){
                        attr_idx++;
                        if(attr_idx < 10) {
                            ll_attr.add(as.toLatexString());
                            String cap = "";
                            for(Mention m : as.getAttributeMentions())
                                if(mentionCapDict.containsKey(m))
                                    cap = mentionCapDict.get(m);
                            ll_attr.add(cap);
                            ll_attr.add("");
                        }
                    }
                }
                FileIO.writeFile(ll_attr, "ex_attr", "txt", true);

                //Minion.export_attachmentCases(docSet);
                System.exit(0);
                /*
                DoubleDict<String> clothFreq = new DoubleDict<>();
                DoubleDict<String> partFreq = new DoubleDict<>();
                for(Document d : docSet){
                    for(Caption c : d.getCaptionList()){
                        List<Mention> agentList = new ArrayList<>();
                        List<Mention> partList = new ArrayList<>();
                        List<Mention> clothList = new ArrayList<>();
                        for(Mention m : c.getMentionList()){
                            if(m.getLexicalType().equals("people") || m.getLexicalType().equals("animals") ||
                               m.getPronounType().isAnimate()){
                                agentList.add(m);
                            } else if(m.getLexicalType().equals("bodyparts")){
                                partList.add(m);
                            } else if(m.getLexicalType().equals("clothing") || m.getLexicalType().equals("colors") ||
                                    m.getLexicalType().equals("clothing/colors")){
                                clothList.add(m);
                            }
                        }
                        List<List<Mention>> agentClusters =
                                ClassifyUtil.collapseMentionListToConstructionList(agentList, c);
                        List<List<Mention>> clothClusters =
                                ClassifyUtil.collapseMentionListToConstructionList(clothList, c);
                        if(partList.size() > 0)
                            partFreq.increment(agentClusters.size() + "|" + partList.size());
                        if(clothClusters.size() > 0)
                            clothFreq.increment(agentClusters.size() + "|" + clothClusters.size());
                    }
                }
                FileIO.writeFile(partFreq, "hist_parts", "csv", true);
                FileIO.writeFile(clothFreq, "hist_cloths", "csv", true);
                System.exit(0);*/

                /*
                Set<String> docIDs = new HashSet<>();
                docIDs.add("4059698218.jpg");
                DBConnector conn = new DBConnector(flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2],flickr30k_mysqlParams[3]);
                docSet = DocumentLoader.getDocumentSet(conn, docIDs);
*/
                /*
                DoubleDict<String> clothHeads = new DoubleDict<>();
                DoubleDict<String> partHeads = new DoubleDict<>();
                for(Document d : docSet){
                    for(Mention m : d.getMentionList()){
                        if(!m.getChainID().equals("0")){
                            if(m.getLexicalType().contains("clothing"))
                                clothHeads.increment(m.getHead().getLemma().toLowerCase());
                            if(m.getLexicalType().contains("bodyparts"))
                                partHeads.increment(m.getHead().getLemma().toLowerCase());
                        }
                    }
                }
                FileIO.writeFile(clothHeads, "hist_clothHead", "csv", false);
                FileIO.writeFile(partHeads, "hist_bodypartHead", "csv", false);
                System.exit(0);*/





            }
        } else if(argList.contains("Data")) {
            String featsFileToConvert = ns.getString("convert_to_arff");
            String featsToExtract = ns.getString("extractFeats");
            String buildDB = ns.getString("buildDB");

            if(featsFileToConvert != null) {
                WekaMulticlass.exportToArff(featsFileToConvert);
            } else if(featsToExtract != null){
                if(featsToExtract.equals("relation"))
                    ClassifyUtil.exportFeatures_relation(docSet, _outroot, numThreads);
                else if(featsToExtract.equals("affinity"))
                    ClassifyUtil.exportFeatures_affinity(docSet, split);
                else if(featsToExtract.equals("nonvis")) {
                    ClassifyUtil.exportFeatures_nonvis(docSet, _outroot);
                } else if(featsToExtract.equals("box_card")) {
                    ClassifyUtil.exportFeatures_boxCard(docSet, _outroot);
                }
            } else if(buildDB != null){
                System.out.println("WARNING: there's a bug where certain cardinalities are null");
                if(buildDB.equals("mysql")){
                    Minion.buildImageCaptionDB(Overlord.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Overlord.flickr30kPath + "RELEASE/", Overlord.flickr30kResources + "img_comments.csv",
                            Overlord.flickr30kResources + "img_crossval.csv", Overlord.flickr30kResources + "img_reviewed.txt",
                            flickr30k_mysqlParams[0], flickr30k_mysqlParams[1], flickr30k_mysqlParams[2], flickr30k_mysqlParams[3]);
                } else if(buildDB.equals("sqlite")){
                    Minion.buildImageCaptionDB(Overlord.flickr30kPath + "Flickr30kEntities_v2.coref",
                            Overlord.flickr30kPath + "RELEASE/", Overlord.flickr30kResources + "img_comments.csv",
                            Overlord.flickr30kResources + "img_crossval.csv", Overlord.flickr30kResources + "img_reviewed.txt",
                            Overlord.flickr30kPath + "Flickr30kEntities_v2_" +Util.getCurrentDateTime("yyyyMMdd") + ".db");
                }
            }
        } else if(argList.contains("Learn")) {

        } else if(argList.contains("Infer")) {
            String nonvisFile = ns.getString("nonvis_scores");
            String relationFile = ns.getString("relation_scores");
            String affinityFile = ns.getString("affinity_scores");
            String cardinalityFile = ns.getString("cardinality_scores");
            String typeCostFile = Overlord.flickr30kResources + "hist_typePairLogProb.csv";

            //Set up the relation inference module
            ILPInference relInf = null;
            switch(ns.getString("inf_type")){
                case "relation": relInf = new ILPInference(docSet, nonvisFile,
                        relationFile, typeCostFile, null, null,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
                case "grounding": relInf = new ILPInference(docSet, nonvisFile,
                        null, null, affinityFile, cardinalityFile,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
                case "joint": relInf = new ILPInference(docSet, nonvisFile,
                        relationFile, typeCostFile, affinityFile, cardinalityFile,
                        ns.getString("graph_root"), ns.getDouble("alpha"));
                    break;
            }

            if(relInf == null){
                Logger.log(new Exception("Could not create relation inference module"));
                System.exit(1);
            }

            //Do inference
            relInf.infer(numThreads, ns.getBoolean("type_constraint"));

            //And evaluate it
            relInf.evaluate(ns.getBoolean("export_files"));
        }
	}

    /**Returns the Namespace object for the given parser, run over
     * the given args; Quits the application if arg parser is violated
     *
     * @param parser
     * @param args
     * @return
     */
    private static Namespace _parseArgs(ArgumentParser parser, String[] args)
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
    private static void _setArgument_flag(Argument arg, String help)
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
    private static void _setArgument_opts(Argument arg, String[] opts,
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
    private static void _setArgument(Argument arg, String help)
    {
        _setArgument(arg, help, null, null, null, null);
    }

    /**Sets up an argument of type with defaultVal and help
     *
     * @param arg
     * @param help
     * @param type
     * @param defaultVal
     */
    private static void _setArgument(Argument arg, String help,
                                     Class type, Object defaultVal)
    {
        _setArgument(arg, help, type, defaultVal, null, null);
    }

    /**Sets up a string argument with specified help and meta-var
     *
     * @param arg
     * @param help
     * @param metavar
     */
    private static void _setArgument(Argument arg, String help, String metavar)
    {
        _setArgument(arg, help, null, null, metavar, null);
    }

    /**Sets up required string argument with specified help and meta-var
     *
     * @param arg
     * @param help
     * @param metavar
     * @param required
     */
    private static void _setArgument(Argument arg, String help, String metavar, Boolean required)
    {
        _setArgument(arg, help, null, null, metavar, required);
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
    private static void _setArgument(Argument arg, String help,
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
