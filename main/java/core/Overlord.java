package core;

import learn.ClassifyUtil;
import learn.WekaMulticlass;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.*;
import net.sourceforge.argparse4j.internal.HelpScreenException;
import net.sourceforge.argparse4j.internal.UnrecognizedArgumentException;
import out.OutTable;
import structures.Caption;
import structures.Document;
import structures.Mention;
import utilities.DBConnector;
import utilities.FileIO;
import utilities.Logger;

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
        //start our runtime clock
		Logger.startClock();

		//arguments for the  manager
		ArgumentParser parser = 
				ArgumentParsers.newArgumentParser("PanOpt");
		parser.defaultHelp(true);
		parser.description("The PanOpt application has several " +
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

        /* Data Group */
        Subparser dataParser = subparsers.addParser("Data");
        setArgument(dataParser.addArgument("--convert_to_arff"),
                "Converts the specified .feats file to .arff format",
                "PATH");
        String[] featOpts = {"pairwise", "affinity"};
        setArgument_opts(dataParser.addArgument("--extractFeats"), featOpts, null,
                "Extracts features to --out");



        /* Learn Group */
        Subparser learnParser = subparsers.addParser("Learn");
        setArgument(learnParser.addArgument("--train_file"), "Training data", "PATH");
        setArgument(learnParser.addArgument("--model_file"), "Model file (save to / load from)", "PATH");
        String[] learners = {"weka_multi", "liblinear_logistic"};
        setArgument_opts(learnParser.addArgument("--learner"), learners, "weka_multi",
                "Specifies which training is taking place");
        setArgument(learnParser.addArgument("batch_size"), "Specifies SIZE batches during training",
                Integer.class, 100, "SIZE", false);
        setArgument(learnParser.addArgument("--epochs"), "Specifies NUM epochs",
                Integer.class, 1000, "NUM", false);
        setArgument(learnParser.addArgument("--eval_file"), "Evaluation data", "PATH");
        setArgument_flag(learnParser.addArgument("--pronom_coref"),
                "Evaluate rule-based pronominal coreference resolution");


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
            } else {
                Map<String, Mention> mentionDict = new HashMap<>();
                Map<String, Document> docDict = new HashMap<>();
                for(Document d : docSet){
                    docDict.put(d.getID(), d);
                    for(Caption c : d.getCaptionList()){
                        for(Mention m : c.getMentionList()){
                            mentionDict.put(m.getUniqueID(), m);
                        }
                    }
                }

                String[] labels = {"null", "coref", "subset", "superset"};
                List<String> mistakes =
                        FileIO.readFile_lineList("/home/ccervan2/source/ImageCaptionLearn_py/pairwise_mistakes.csv");
                OutTable ot = new OutTable("Gold", "Predicted", "m_1", "m_1_type", "m_2", "m_2_type", "Caption_1", "Caption_2");
                for(int i=1; i<mistakes.size(); i++){
                    String row = mistakes.get(i).trim();
                    if(!row.isEmpty()){
                        String[] cells = row.split(",");
                        int gold = Integer.parseInt(cells[0]);
                        int pred = Integer.parseInt(cells[1]);
                        String mention_1 = cells[2];
                        String mention_2 = cells[3];
                        Mention m1 = mentionDict.get(mention_1);
                        Mention m2 = mentionDict.get(mention_2);
                        Caption c1 = docDict.get(m1.getDocID()).getCaption(m1.getCaptionIdx());
                        Caption c2 = docDict.get(m2.getDocID()).getCaption(m2.getCaptionIdx());
                        ot.addRow(labels[gold], labels[pred], m1.toString(),
                                m1.getLexicalType(), m2.toString(), m2.getLexicalType(),
                                c1.toString(), c2.toString());
                    }
                }
                ot.writeToCsv("ex_pairwise_mistakes", false);
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
                    ClassifyUtil.exportFeatures_affinity(docSet, _outroot, numThreads);
            }

        } else if(argList.contains("Learn")) {

        }
	}



	/**This is just quick and dirty code for when I need to
	 * do something that won't really persist. This function
	 * may as well say 'deleteme' on it.
	 */
	/*
	private static void debug(String... args)
    {
        if(args[0].equals("data")){
            Collection<Document> docSet = DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath));
            List<String> ll = new ArrayList<>();
            for(Document d : docSet){
                for(Caption c : d.getCaptionList()){
                    List<String> lemStr = new ArrayList<>();
                    for(Token t : c.getTokenList())
                        lemStr.add(t.getLemma());
                    ll.add(d.getID() + "#" + c.getIdx() + "\t" + StringUtil.listToString(lemStr, " "));
                }
            }
            FileIO.writeFile(ll, "Flickr30kEntities_v2", "lemma", false);
            //WekaMulticlass.exportToArff("../flickr30kEntities_v2_pairwise_train.feats");
            //WekaMulticlass.exportToArff("../flickr30kEntities_v2_pairwise_dev.feats");
        } else if(args[0].equals("train")){
            WekaMulticlass wekaMulticlass = new WekaMulticlass();
            Logger.log("Training classifier");
            //wekaMulticlass._train("../flickr30kEntities_v2_pairwise_train.arff",
            //        "models/pairwise_log.model");
            //wekaMulticlass.train("/Users/syphonnihil/source/working/flickr30kEntities_v2_pairwise_dev_with_IDs.arff", "pairwise_mcc.model");
            wekaMulticlass.train("../flickr30kEntities_v2_pairwise_train.arff",
                    "models/pairwise_mcc.model", 1000, 100000, true);
        } else if(args[0].equals("eval")){
            Logger.log("Loading model from file");
            WekaMulticlass wekaMulticlass = new WekaMulticlass("models/pairwise_mcc.model");
            Logger.log("Evaluating classifier");
            wekaMulticlass.evaluate("../flickr30kEntities_v2_pairwise_dev.arff");
            /*wekaMulticlass._eval("../flickr30kEntities_v2_pairwise_dev.arff",
                    "models/pairwise_log.model");*
        } else if(args[0].equals("token")){
            Set<String[]> mismatchCaps = new HashSet<>();

            Logger.log("Reading token file");
            Map<String, String> capDict_tok = new HashMap<>();
            List<String> capIDs = new ArrayList<>();
            for(String line : FileIO.readFile_lineList("../results_20130124_fixed.token")){
                String[] lineParts = line.split("\t");
                capDict_tok.put(lineParts[0], lineParts[1]);
                capIDs.add(lineParts[0]);
            }

            Logger.log("Loading documents from Entities (v2) DB");
            Map<String, Document> docDict_v2 = new HashMap<>();
            Map<String, String> capDict_v2 = new HashMap<>();
            for(Document d : DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath))){
                docDict_v2.put(d.getID(), d);
                for(Caption c : d.getCaptionList()){
                    String capID = d.getID() + "#" + c.getIdx();
                    String capStr_tok = capDict_tok.get(capID);
                    String capStr = c.toString();
                    if(!capStr.equals(capStr_tok))
                        mismatchCaps.add(new String[]{capStr, capStr_tok});
                    capDict_v2.put(capID, c.toString());
                }
            }

            List<String> ll_tokV2 = new ArrayList<>();
            for(String capID : capIDs)
                ll_tokV2.add(capID + "\t" + capDict_v2.get(capID));
            FileIO.writeFile(ll_tokV2, "../flickr30kEntities_v2", "token", false);

        } else if(args[0].equals("pronom")){
            Collection<Document> docSet = DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath), 0);
            //Minion.exportPronomAttrEx(docSet);
            Minion.pronomCorefEval(docSet);
        } else if(args[0].equals("legacy")){
            List<String> typoCaps = new ArrayList<>();
            Set<legacy.Document> docSet = legacy.Page.getDocumentSet();
            for(legacy.Document d : docSet){
                for(legacy.Caption c : d.getCaptionArr()){
                    if(c.TYPO)
                        typoCaps.add(d.getID() + ".jpg#" + c.getIdx());
                }
            }
        } else if(args[0].equals("box")) {
            Collection<Document> docSet = DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath), 1);
            DoubleDict<Boolean> distro = new DoubleDict<>();
            for(Document d : docSet){
                for(BoundingBox b : d.getBoundingBoxSet()){
                    Set<Mention> mentionBoxSet = d.getMentionSetForBox(b);
                    for(Mention m : d.getMentionList()){
                        distro.increment(mentionBoxSet.contains(m));
                    }
                }
            }
            System.out.println("True distro");
            for(Boolean label : distro.keySet())
                System.out.printf("%s : %.2f%%\n", label, 100.0 * distro.get(label) / distro.getSum());

        } else if(args[0].equals("subset")){
            Collection<Document> docSet = DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath), 0);

        } else if(args[0].equals("feats")){
            Collection<Document> docSet = DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath), 1);
            ClassifyUtil.initLists();
            ClassifyUtil.exportFeatures_pairwise(docSet, "../flickr30kEntities_v2_pairwise_train", 24);
        }
        System.exit(0);




    }
*/


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
