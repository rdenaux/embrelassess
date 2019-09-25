package net.expertsystem.embrelassess;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Optional;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;

import net.expertsystem.util.TimeUtil;
import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.IndexWord;
import net.sf.extjwnl.data.IndexWordSet;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.Pointer;
import net.sf.extjwnl.data.Synset;
import net.sf.extjwnl.dictionary.Dictionary;

/**
 * Analyses a vecsigrafo vocab (and a WordNet for the same language) and generates files with pairs of vocabulary terms that 
 * are related according to WordNet. These pairs are meant to be used for checking whether the embedding space encodes 
 * these relations somehow (a separate class performs this check).
 *   
 * @author rdenaux
 *
 */
public class WNRelationPairExtractor {

	private static final Logger log = LoggerFactory.getLogger(WNRelationPairExtractor.class);
	
	public final String synset_POS_Offset_PatternStr = "(wn31_)([A-Z]+)#(\\d+)";
	private final Pattern synset_POS_Offset_Pattern = Pattern.compile(synset_POS_Offset_PatternStr);

	static enum Opt {
		vocab(File.class),
		out_dir(File.class)
		;
		
		private final Class<?> clz;
		private final Optional<?> defVal;
		
		private <T> Opt(Class<T> clz, Optional<T> defVal) {
			this.clz = clz;
			this.defVal = defVal;
		}
		
		private <T> Opt(Class<T> clz) {
			this(clz, Optional.<T>absent());
		}
		
		private Opt(int intDefVal) {
			this(Integer.class, Optional.of(Integer.valueOf(intDefVal)));
		}
		
		public Integer getInt(CommandLine cmd) {
			if (Integer.class != clz) throw new IllegalArgumentException("Cannot get int for " + this);
			if (cmd.hasOption(name())) {
				return Integer.parseInt(cmd.getOptionValue(name()));
			} else if (defVal.isPresent()) { 
				return (Integer)defVal.get();
			} else throw new RuntimeException("No argument passed (and no default) for " + this);
		}

		public File getFile(CommandLine cmd) {
			if (File.class != clz) throw new IllegalArgumentException("Cannot get File for " + this);
			if (cmd.hasOption(name())) {
				return new File(cmd.getOptionValue(name()));
			} else if (defVal.isPresent()) {
				return (File)defVal.get();
			} else throw new RuntimeException("No argument passed (and no default) for " + this);
		}
		
		public String getString(CommandLine cmd) {
			if (String.class != clz) throw new IllegalArgumentException("Cannot get Value for " + this);
			if (cmd.hasOption(name())) {
				return cmd.getOptionValue(name());
			} else if (defVal.isPresent()) {
				return (String)defVal.get();
			} else throw new RuntimeException("No argument passed (and no default) for " + this);
		}
		
	}
	
	
	private static enum Lang {
		ENGLISH("en");
		
		private Lang(String iso2code) {
			this.iso2Code = iso2code;
		}
		
		/**
		 * The ISO-639-1 two-letter code for the language.
		 */
		public final String iso2Code;
	}
		
	private final Dictionary dict;
	private final List<String> vocab;
	private final Lang lang;
	private final Set<Synset> synVoc; 
	private final Set<String> lemVoc; 
	private final Random rnd = new Random();
		
	public static void main(String[] args) {
		Options options = getCommandLineOptions();
		CommandLineParser parser = new GnuParser(); //new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		final long start = System.currentTimeMillis();
		try {
			CommandLine cmd = parser.parse(options, args);
			final List<String> vocab = Files.readAllLines(Opt.vocab.getFile(cmd).toPath());
			final Dictionary dict = Dictionary.getDefaultResourceInstance();
			WNRelationPairExtractor app = new WNRelationPairExtractor(dict, vocab, Lang.ENGLISH);
			Path outDir = cmd.hasOption(Opt.out_dir.name()) ? Opt.out_dir.getFile(cmd).toPath() : 
				Paths.get(Opt.vocab.getFile(cmd).getParent(), "wn_vocabrels");
			@SuppressWarnings("rawtypes")
			Multimap<String, Pair> synPairs = app.extractRelations(outDir);
			app.writeFiles(synPairs, outDir);
//			System.out.println(String.format("Dir %s, filename %s", dir, simFilePattern.getName()));
		} catch (ParseException e) {
			log.error("Error reading configuration value", e);
			final String header = "Provides an interface for querying for the topN nearest points in a vector space.";
			final String footer = "2017 Expert System Iberia"; 
			formatter.printHelp(WNRelationPairExtractor.class.getSimpleName(), header, options, footer);

			System.exit(1);
			return;
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Finished in " + TimeUtil.humanTimeFrom(start));
			System.exit(1);
		}
		System.out.println("Finished in " + TimeUtil.humanTimeFrom(start));
		System.exit(0);
		
	}
	
	public WNRelationPairExtractor(Dictionary dict, List<String> vocab, Lang lang) {
		super();
		this.dict = dict;
		this.vocab = vocab;
		this.lang = lang;
		synVoc = extractSynsets(vocab);
		lemVoc = extractLemmas(vocab);
	}

	private static double negSamplesFactor = 1.0;
	private static double negSwitchFactor = 1.0;
	
	/**
	 * Writes the collected pairs to a file in outDir. Notice that the input pairs only contain positive examples and 
	 * this method generates negative samples based on {@link #negSamplesFactor} and {@link #negSwitchFactor}.
	 * 
	 * @see #generateNegativeSamples(Multimap, String, double, double)
	 * 
	 * @param pairs a Multimap from relation-name to collections of <b>positive</b> pairs.
	 * @param outDir
	 * @throws IOException
	 */
	private void writeFiles(Multimap<String, Pair> pairs, Path outDir) throws IOException {
		Files.createDirectories(outDir);
		for (String relName: pairs.keySet()) {
			final String fName = relName.replace(" ", "_").replace("/", "-");
			final Path oPath = Paths.get(outDir.toString(), String.format("%s__%s.txt", fName, pairs.get(relName).size()));
			try {
				writeRels(oPath, pairs.get(relName), generateNegativeSamples(pairs, relName, negSamplesFactor, negSwitchFactor));
			} catch (IOException e) {
				log.error("Error writing relations for " + relName, e);
			}
		}
		//TODO: write tsv with relnames and counts
	}

	/**
	 * Generate negative samples for a given relation. The number of generated samples is controlled by two factors.
	 * 
	 * @param pairs the whole set of positive pairs (for <code>relName</code> and for other relations)
	 * @param relName the specific relation for which to generate negative samples
	 * @param targetFactor specifies how many negative samples to generate as a factor of the number of positive pairs
	 * 	for <code>relName</code>. This method will try to generate <code> n * targetFactor</code> negative pairs, where 
	 *  <code>n</code> is the number of positive pairs for <code>relName</code>
	 * @param negSwitchedFactor specifies how many negative samples we should aim to generate by using <b>negative switching</b>
	 * 	ie by switching existing pairs for the relation. If the set of positive pairs is not sufficient for generating this 
	 *  desired number, alternative strategies will be used to generate negative pairs.
	 * @return a SetMultimap of generated negative pairs for <code>relName</code> the key is the name of the strategy 
	 * 	used to generate the negative pair, and the values are sets of negative pairs. Currently, strategies can be: 
	 *  <i>NegSwitched</i>, <i>Compatible-rel</i> and <i>Converted</i>.
	 */
	private SetMultimap<String, Pair> generateNegativeSamples(Multimap<String, Pair> pairs, String relName, double targetFactor, double negSwitchedFactor) {
		assert(negSwitchedFactor <= targetFactor);
		Set<Pair> positives = ImmutableSet.copyOf(pairs.get(relName));
		final int targetNegSwitchedSize = (int) Math.floor(positives.size() * negSwitchedFactor);
		final int targetSize = (int) Math.floor(positives.size() * targetFactor);
		SetMultimap<String, Pair> result = HashMultimap.create();
		
		Optional<Class<? extends Pair>> posType = determineType(positives);
		
		//add negative switched
		Set<Object> posFroms = new HashSet<>();
		Set<Object> posTos = new HashSet<>();
		for (Pair<?,?> posPair: positives) {
			posFroms.add(posPair.from());
			posTos.add(posPair.to());
		}
		List<Object> fromPool = ImmutableList.copyOf(posFroms);
		List<Object> toPool = ImmutableList.copyOf(posTos);
		for (int i = 0; i < targetNegSwitchedSize * 3; i++) {
			Pair candidatePair = randomPair(fromPool, toPool, posType);
			if (positives.contains(candidatePair)) continue;
			result.put("NegSwitched", candidatePair);
			if (result.values().size() >= targetNegSwitchedSize) break;
		}
		
		if (result.values().size() < targetNegSwitchedSize) log.warn(String.format(
				"Failed to generate a full set of switched negatives for %s, only generated %s of %s", 
				relName, result.values().size(), targetNegSwitchedSize));
		
		//add compatible relations from the pool of positive pairs
		List<Pair> allPairs = new ArrayList<Pair>(pairs.values());
		Collections.shuffle(allPairs);
		for (Pair sp: allPairs) {
			if (positives.contains(sp)) continue; //make sure it is not a positive pair
			if (matchesType(posType, sp)) {
				result.put("Compatible-rel", sp);
			}
			if (result.values().size() >= targetSize) break;
		}
		
		
		if (result.values().size() < targetSize) { //not balanced yet, try modifying existing relations
			for (Pair sp: allPairs) {
				if (positives.contains(sp)) continue; //make sure it is not a positive pair
				Optional<?> optConverted = convertToType(posType, sp);
				if (optConverted.isPresent()) {
					Pair converted = (Pair)optConverted.get();
					if (!positives.contains(converted)) result.put("Converted", converted);
				}
				if (result.size() >= positives.size()) break;
			}
		}
		
		if (result.size() < positives.size()) 
			log.warn(String.format("Failed to generate a full set of negatives for %s, only generated %s of %s", 
					relName, result.values().size(), targetSize));
		return result;
	}

	/**
	 * Generates a random pair of posType using a pool of froms and tos.
	 * 
	 * @param fromPool
	 * @param toPool
	 * @param posType
	 * @return
	 */
	@SuppressWarnings("rawtypes")
	private Pair randomPair(List<Object> fromPool, List<Object> toPool, Optional<Class<? extends Pair>> targetType) {
		if (!targetType.isPresent()) throw new RuntimeException("Cannot generate random pair without a type");
		if (targetType.get().equals(LemPair.class)) {
			return new LemPair((String)selectRandom(fromPool), (String)selectRandom(toPool));
		} else if (targetType.get().equals(SynsetPair.class)) {
			return new SynsetPair((Synset)selectRandom(fromPool), (Synset)selectRandom(toPool));
		} else if (targetType.get().equals(LemPOSPair.class)) {
			return new LemPOSPair((String)selectRandom(fromPool), (POS)selectRandom(toPool));
		} else if (targetType.get().equals(SynPOSPair.class)) {
			return new SynPOSPair((Synset)selectRandom(fromPool), (POS)selectRandom(toPool));
		} else if (targetType.get().equals(LemSynsetPair.class)) {
			return new LemSynsetPair((String)selectRandom(fromPool), (Synset)selectRandom(toPool));
		}
		throw new RuntimeException("Unsupported targetType" + targetType.get());
	}

	private <T> T selectRandom(List<T> pool) {
		final int id = rnd.nextInt(pool.size());
		return pool.get(id);
	}	
	
	private <T> T selectRandom(Set<T> pool) {
		final int id = rnd.nextInt(pool.size());
		int cnt = 0;
		for (T obj: pool) {
			if (id == cnt) return obj;  
			cnt++;
		}
		throw new RuntimeException(String.format("Should have chosen id within range %d, but chose %d", pool.size(), id));
	}

	@SuppressWarnings("rawtypes")
	private Optional<Pair> convertToType(Optional<Class<? extends Pair>> targetType, Pair candidate) {
		if (!targetType.isPresent()) return Optional.absent();
		// in case we introduce synDomPair or lemDomPair
		return Optional.absent();
	}

	
	private boolean canBeConvertedToType(Optional<Class<? extends Pair>> targetType, Pair candidate) {
		if (!targetType.isPresent()) return false;
		//return (targetType.equals(SynDomPair.class) && candidate.getClass().equals(LemSynPair.class));
		return false;
	}

	@SuppressWarnings("rawtypes")
	private boolean matchesType(Optional<Class<? extends Pair>> posType, Pair sp) {
		if (!posType.isPresent()) return true;
		return posType.get().equals(sp.getClass());
	}

	@SuppressWarnings("rawtypes")
	private Optional<Class<? extends Pair>> determineType(Set<Pair> pairs) {
		if (pairs.isEmpty()) return Optional.absent();
		Class<? extends Pair> clz = pairs.iterator().next().getClass();
		for (Pair p: pairs) {
			if (!clz.equals(p.getClass())) return Optional.absent(); //multiclass
		}
		return Optional.of(clz);
	}

	@SuppressWarnings("rawtypes")
	private void writeRels(Path oPath, Collection<Pair> positives, SetMultimap<String, Pair> negatives) throws IOException {
		final boolean append = false;
		try (
			BufferedWriter bwt = new BufferedWriter(new FileWriter(oPath.toFile(), append));				
		) {
			Iterator<Pair> posit = positives.iterator();
			SetMultimap<Pair, String> negToTypes = reverse(negatives);
			Iterator<Pair> negit = negToTypes.keySet().iterator();
			while(posit.hasNext() || negit.hasNext()) {
				if (posit.hasNext()) {
					Pair spair = posit.next();
					bwt.append(String.format("%s\t%s\t%s\t%s\n", encodeTerm(spair.from()), encodeTerm(spair.to()), 1, "positive"));
				}
				if (negit.hasNext()) {
					Pair spair = negit.next();
					bwt.append(String.format("%s\t%s\t%s\t%s\n", encodeTerm(spair.from()), encodeTerm(spair.to()), 0, negToTypes.get(spair).toString()));
				}
			}
			bwt.flush();
		}
	}

	private String encodeTerm(Object term) {
		if (term instanceof Synset) { //synset
			Synset syn = (Synset)term;
			// synsets are encoded as wn31_<pos_name>#<offset> which is unique for a version of wordNet
			// an alternative encoding could be wn31_<main_lemma>.<pos_key>.<pos_lem_index> but this is not unique (ie. 
			//  several such encodings can refer to the same synset
			return String.format("wn31_%s#%s", syn.getPOS().name(), syn.getOffset());
		} else if (term instanceof POS) {
			return ((POS)term).name();
		} else if (term instanceof String) {// lemma
			return (String)term;
		} else throw new IllegalArgumentException("term should be Integer for a syncon or String for a lemma");
	}

	/**
	 * Generic pair of elements to be included in a KG-derived dataset
	 * @author rdenaux
	 *
	 * @param <F> from type
	 * @param <T> to type
	 */
	static interface Pair<F,T> {
		F from();
		T to();
	}
	
	/**
	 * Pair from a Lemma to a Synset
	 * @author rdenaux
	 *
	 */
	static class LemSynsetPair implements Pair<String, Synset> {
		String lemma;
		Synset syn;
		public LemSynsetPair(String lemma, Synset syn) {
			super();
			this.lemma = lemma;
			this.syn = syn;
		}
		public String from() {
			return lemma;
		}
		
		public Synset to() {
			return syn;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((lemma == null) ? 0 : lemma.hashCode());
			result = prime * result + syn.hashCode();
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			LemSynsetPair other = (LemSynsetPair) obj;
			if (lemma == null) {
				if (other.lemma != null)
					return false;
			} else if (!lemma.equals(other.lemma))
				return false;
			if (syn != other.syn)
				return false;
			return true;
		}
	}
	
	/**
	 * Pair between two lemmas
	 * @author rdenaux
	 *
	 */
	static class LemPair implements Pair<String, String> {
		String parent;
		String child;
		public LemPair(String parent, String child) {
			super();
			this.parent = parent;
			this.child = child;
		}
		public String from() {
			return parent;
		}
		public String to() {
			return child;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((child == null) ? 0 : child.hashCode());
			result = prime * result + ((parent == null) ? 0 : parent.hashCode());
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			LemPair other = (LemPair) obj;
			if (child == null) {
				if (other.child != null)
					return false;
			} else if (!child.equals(other.child))
				return false;
			if (parent == null) {
				if (other.parent != null)
					return false;
			} else if (!parent.equals(other.parent))
				return false;
			return true;
		}
		
	}

	/**
	 * Pair between two synsets
	 * @author rdenaux
	 *
	 */
	static class SynsetPair implements Pair<Synset, Synset> {
		Synset parent;
		Synset child;
		public SynsetPair(Synset parent, Synset child) {
			super();
			this.parent = parent;
			this.child = child;
		}
		public Synset from() {
			return parent;
		}
		public Synset to() {
			return child;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((child == null) ? 0 : child.hashCode());
			result = prime * result + ((parent == null) ? 0 : parent.hashCode());
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			SynsetPair other = (SynsetPair) obj;
			if (child == null) {
				if (other.child != null)
					return false;
			} else if (!child.equals(other.child))
				return false;
			if (parent == null) {
				if (other.parent != null)
					return false;
			} else if (!parent.equals(other.parent))
				return false;
			return true;
		}
	}
		
	/**
	 * Pair from a synset to a part-of-speech
	 * @author rdenaux
	 *
	 */
	static class SynPOSPair implements Pair<Synset, POS> {
		Synset syn;
		POS pos;
		public SynPOSPair(Synset syn, POS pos) {
			super();
			this.syn = syn;
			this.pos = pos;
		}
		@Override
		public Synset from() {
			return syn;
		}
		@Override
		public POS to() {
			return pos;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((pos == null) ? 0 : pos.hashCode());
			result = prime * result + syn.hashCode();
			return result;
		}
		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			SynPOSPair other = (SynPOSPair) obj;
			if (pos != other.pos)
				return false;
			if (syn != other.syn)
				return false;
			return true;
		}
	}
	
	/**
	 * Pair from a lemma to a part-of-speech
	 * @author rdenaux
	 *
	 */
	static class LemPOSPair implements Pair<String, POS> {
		String lem;
		POS pos;

		public LemPOSPair(String lem, POS pos) {
			super();
			this.lem = lem;
			this.pos = pos;
		}

		@Override
		public String from() {
			return lem;
		}

		@Override
		public POS to() {
			return pos;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((lem == null) ? 0 : lem.hashCode());
			result = prime * result + ((pos == null) ? 0 : pos.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			LemPOSPair other = (LemPOSPair) obj;
			if (lem == null) {
				if (other.lem != null)
					return false;
			} else if (!lem.equals(other.lem))
				return false;
			if (pos != other.pos)
				return false;
			return true;
		}
	}
	
	/**
	 * Main extraction method
	 * @param outDir
	 * @return
	 */
	private Multimap<String, Pair> extractRelations(Path outDir) {
		log.info(String.format("Extracting relations from %s synsets and %s lemmas", synVoc.size(), lemVoc.size()));
		
		final int maxDepth = 1;
		final ListMultimap<String, SynsetPair> synPairs = extractSynsetPairs(synVoc, maxDepth);
		final ListMultimap<String, SynsetPair> randSynPairs = extractRandomSynPairs();
//		final ListMultimap<String, SynDomPair> syndomPairs = extractSynDomPairs(synVoc, domVoc);
		final ListMultimap<String, SynPOSPair> synposPairs = extractSynPOSPairs(synVoc);
		final SetMultimap<String, Synset> lemToSyn = extractLemToSynInVoc(lemVoc);
		writePairsStats(synPairs, Paths.get(outDir.toString(), "synpairs.tsv"));
		final Multimap<String, LemPair> lemPairs = toLemPairs(synPairs, lemToSyn);
//		final ListMultimap<String, SynsetPair> synPairs = extractSynsetPairs(ImmutableSet.copyOf(lemToSyn.values()), maxDepth);
		writePairsStats(lemPairs, Paths.get(outDir.toString(), "lempairs.tsv"));
		final ListMultimap<String, LemPair> randLemPairs = extractRandomLemPairs();
		final Multimap<String, LemPOSPair> lemposPairs = toLemPosPairs(synposPairs, lemToSyn);
//		final Multimap<String, LemDomPair> lemdomPairs = toLemDomPairs(syndomPairs, lemToSyn);
		
		ListMultimap<String, Pair> result = ArrayListMultimap.create();
		result.putAll(synPairs);
		result.putAll(randSynPairs);
//		result.putAll(syndomPairs);
		result.putAll(lemPairs);
		result.putAll(randLemPairs);
		result.putAll(toLemToSynPairs(lemToSyn, synPairs)); 
		result.putAll(synposPairs);
		result.putAll(lemposPairs);
//		result.putAll(lemdomPairs);
		return result;
	}
	
	private Multimap<String, LemSynsetPair> toLemToSynPairs(SetMultimap<String, Synset> lemToSyn,
			ListMultimap<String, SynsetPair> synPairs) {
		SetMultimap<Synset, String> synToLem = reverse(lemToSyn);
		SetMultimap<String, LemSynsetPair> result = HashMultimap.create();
		for (String rel: synPairs.keySet()) {
			final String lemsynRel = rel.replace("syn2syn_", "lem2syn_");
			for (SynsetPair spair : synPairs.get(rel)) {
				for (String parLem: synToLem.get(spair.parent)) {
					result.put(lemsynRel, new LemSynsetPair(parLem, spair.child));
				}
			}
		}
		log.info(String.format("Converted to %s lemsyn-pairs for %s relations in vocab.",
				result.size(), result.keySet().size()));
		result.putAll(asLemToSynPairs(lemToSyn)); //lemma relations
		return result;
	}

	private Multimap<String, LemPOSPair> toLemPosPairs(ListMultimap<String, SynPOSPair> synPosPairs,
			SetMultimap<String, Synset> lemToSyn) {
		SetMultimap<Synset, String> synToLem = reverse(lemToSyn);
		SetMultimap<String, LemPOSPair> result = HashMultimap.create();
		for (String rel: synPosPairs.keySet()) {
			final String lemRel = rel.replace("syn2pos_", "lem2pos_");
			for (SynPOSPair spair : synPosPairs.get(rel)) {
				for (String parLem: synToLem.get(spair.from())) {
					POS chiPOS = spair.to();
					if (parLem.equals(chiPOS)) continue; //ignore 
					result.put(lemRel, new LemPOSPair(parLem, chiPOS));
				}
			}
		}
		return result;
	}
	
	private ListMultimap<String, SynPOSPair> extractSynPOSPairs(Set<Synset> synVoc2) {
		ListMultimap<String, SynPOSPair> result = ArrayListMultimap.create();
		int missingCnt = 0;
		for (Synset syn: synVoc2) {
			POS pos = syn.getPOS();
			if (pos == null) {
				missingCnt++;
				continue;
			}
			
			result.put(String.format("syn2pos_%s", pos.name()), new SynPOSPair(syn, pos));
		}
		log.info(String.format("Found %s synPOS relations in vocab. %s syncons missing a POS. %s", 
				result.size(), missingCnt, result.keySet().stream()
					.map(rel -> String.format("%s x %s", rel, result.get(rel).size()))
					.collect(Collectors.joining())
				));
		return result;
	}

	private ListMultimap<String, SynsetPair> extractRandomSynPairs() {
		List<Integer> sizes = ImmutableList.of(50, 100, 200, 500, 1000, 5000, 10000, 50000);
		ListMultimap<String, SynsetPair> result = ArrayListMultimap.create();
		for (int size : sizes) {
			if (size > (synVoc.size() * synVoc.size())/2) {
				log.info(String.format("Not generating random syn2syn pairs for size %d since vocab is not big enough", size));
			}
			Set<SynsetPair> pairs = new HashSet<>();
			for (int i = 0; i < size; i++) {
				pairs.add(randomSynPair());
			}
			result.putAll(String.format("syn2syn_random_%d", size),  pairs); 
		}
		return result;
	}

	private SynsetPair randomSynPair() {
		return new SynsetPair(randomElement(synVoc), randomElement(synVoc));
	}

	
	private <S,T, P extends Pair<S,T>> void writePairsStats(Multimap<String, P> pairs, Path outPath) {
		try {
			com.google.common.io.Files.createParentDirs(outPath.toFile());
		} catch (IOException e) {
			log.error("", e);
		}
		CSVFormat tsvFormat = CSVFormat.TDF.withHeader("rel", "posCnt", "totVocab", "srcVocab", "tgtVocab", "sharedVocab");
		try ( 
				Writer out = Files.newBufferedWriter(outPath, StandardCharsets.UTF_8);
				CSVPrinter vecPrinter = new CSVPrinter(out, tsvFormat);
		) {
			for (String rel: pairs.keySet()) {
				Set<S> srcs = new HashSet<>();
				Set<T> tgts = new HashSet<>();
				final int relCnt = pairs.get(rel).size();
				for (Pair<S,T> pair: pairs.get(rel)) {
					srcs.add(pair.from());
					tgts.add(pair.to());
				}
				final int totVocab = Sets.union(srcs, tgts).size();
				final int interVocab = Sets.intersection(srcs, tgts).size();
				vecPrinter.printRecord(rel, relCnt, totVocab, srcs.size(), tgts.size(), interVocab);
			}
		} catch (IOException e) {
			log.error("Error writing stats for synset pairs", e);
		}
	}

	/*
	private Multimap<String, LemPOSPair> toLemPosPairs(ListMultimap<String, SynPOSPair> synPosPairs, SetMultimap<String, Integer> lemToSyn) {
		SetMultimap<Integer, String> synToLem = reverse(lemToSyn);
		SetMultimap<String, LemPOSPair> result = HashMultimap.create();
		for (String rel: synPosPairs.keySet()) {
			final String lemRel = rel.replace("syn2pos_", "lem2pos_");
			for (SynPOSPair spair : synPosPairs.get(rel)) {
				for (String parLem: synToLem.get(spair.from())) {
					Grammar chiPOS = spair.to();
					if (parLem.equals(chiPOS)) continue; //ignore 
					result.put(lemRel, new LemPOSPair(parLem, chiPOS));
				}
			}
		}
		return result;
	}*/
	
//	private Multimap<String, LemDomPair> toLemDomPairs(ListMultimap<String, SynDomPair> syndomPairs,
//			SetMultimap<String, Integer> lemToSyn) {
//		SetMultimap<Integer, String> synToLem = reverse(lemToSyn);
//		SetMultimap<String, LemDomPair> result = HashMultimap.create();
//		for (String rel: syndomPairs.keySet()) {
//			final String lemRel = rel.replace("syn2lem_", "lem2lem_");
//			for (SynDomPair spair : syndomPairs.get(rel)) {
//				for (String parLem: synToLem.get(spair.from())) {
//					String chidom = spair.to();
//					if (parLem.equals(chidom)) continue; //ignore 
//					result.put(lemRel, new LemDomPair(parLem, chidom));
//				}
//			}
//		}
//		return result;
//	}
	
	/**
	 * Convert a set of {@link SynsetPair}s to equivalent {@link LemPair}s by using the lemma to synset relations
	 * previously extracted.
	 * 
	 * @param synPairs the pairs to convert
	 * @param lemToSyn relates lemmas to their synsets
	 * @return a version of <code>synPairs</code> where all synsets have been replaced by their lemmas
	 */
	private Multimap<String, LemPair> toLemPairs(
			ListMultimap<String,SynsetPair> synPairs, SetMultimap<String, Synset> lemToSyn) {
		SetMultimap<Synset, String> synToLem = reverse(lemToSyn);
		SetMultimap<String, LemPair> result = HashMultimap.create();
		for (String rel: synPairs.keySet()) {
			final String lemRel = rel.replace("syn2syn_", "lem2lem_");
			for (SynsetPair spair : synPairs.get(rel)) {
				for (String parLem: synToLem.get(spair.parent)) {
					for (String chiLem: synToLem.get(spair.child)) {
						if (parLem.equals(chiLem)) continue; //ignore 
						result.put(lemRel, new LemPair(parLem, chiLem));
					}
				}
			}
		}
		log.info(String.format("Found %s lem-pairs for %s relations in vocab.",
				result.size(), result.keySet().size()));
//		log.info(String.format("Syn-pair counts: %s", relFoundCnt));
		for (Synset syn: synToLem.keySet()) {
			for (String lemPar: synToLem.get(syn)) {
				for (String lemChi: synToLem.get(syn)) {
					if (lemPar.equals(lemChi)) continue;
					result.put("lem2lem_synonym", new LemPair(lemPar, lemChi));
				}
			}
		}
		return result;
	}

	private <A, B> SetMultimap<B, A> reverse(SetMultimap<A, B> aToB) {
		SetMultimap<B, A> result = HashMultimap.create();
		for (A a: aToB.keySet()){
			for (B b: aToB.get(a)) {
				result.put(b, a);
			}
		}
		return result;
	}

	private Multimap<String, LemSynsetPair> asLemToSynPairs(SetMultimap<String, Synset> lemToSyn) {
		SetMultimap<String, LemSynsetPair> result = HashMultimap.create();
		for (String lem: lemToSyn.keySet()) {
			for (Synset syn: lemToSyn.get(lem)) {
				result.put("lem2syn_lemma", new LemSynsetPair(lem, syn));
			}
		}
		log.info(String.format("Found %s lemsyn-pairs for %s relation(s) in vocab.",
				result.size(), result.keySet().size()));
		return result;
	}

	private SetMultimap<String, Synset> extractLemToSynInVoc(Set<String> lemVoc) {
		SetMultimap<String, Synset> result = HashMultimap.create();
		for (String lem: lemVoc) {
			IndexWordSet iws = getAllIndexWords(lem);
			for (IndexWord iw: iws.getIndexWordArray()) {
				result.putAll(lem, iw.getSenses());
			}
		}
		return result;
	}
	
	private IndexWordSet getAllIndexWords(String lem) {
		// do not use dict.lookupAllIndexWords(lem), it's too slow
        final String lemma = lem.trim().toLowerCase();
        IndexWordSet set = new IndexWordSet(lemma);
        for (POS pos : POS.getAllPOS()) {
        	try {
        		IndexWord current = dict.getIndexWord(pos, lemma);
        		if (current != null) {
        			set.add(current);
        		}
        	} catch (JWNLException e) {
        		log.error("Failed to get index word for " + lemma, e);
        	}
        }
        return set;
	}

	private ListMultimap<String, SynsetPair> extractSynsetPairs(Set<Synset> synVoc, int maxDepth) {
		return extractSynsetPairs(synVoc, maxDepth, 0);
	}
	
	
	private ListMultimap<String, SynsetPair> extractSynsetPairs(Set<Synset> synVoc, final int maxDepth, final int minDepth) {
		assert(minDepth >= 0);
		assert(maxDepth > minDepth);
		Multiset<String> relFoundCnt = HashMultiset.create();
		ListMultimap<String, SynsetPair> synPairs = ArrayListMultimap.create();
		Multiset<String> totrelCnt = HashMultiset.create();
		for (Synset synset: synVoc) {
			for (Pointer pointer : synset.getPointers()) {
				final String relName = pointer.getType().getLabel();
				
//				final Set<Integer> ignoreChildren = minDepth > 0 ? pointer.getTargetSynset() : ImmutableSet.of();
				
				try {
					Synset child = pointer.getTargetSynset();
					if (synVoc.contains(child)) {
						relFoundCnt.add(relName);
						totrelCnt.add(relName);
						synPairs.put("syn2syn_"+relName, new SynsetPair(synset, child));
					}
				} catch (JWNLException e) {
					log.error("Failed to get target for " + relName, e);
				}
			}
		}

		log.info(String.format("Found %s syn-pairs for %s relations in vocab, out of %s pairs in wordNet. I.e. %.3f%% coverage",
				relFoundCnt.size(), relFoundCnt.elementSet().size(), totrelCnt.size(), 
				100.0 * relFoundCnt.size() / (totrelCnt.size() + 0.001)));
		log.info(String.format("Syn-pair counts: %s", relFoundCnt));
		return synPairs;
	}

	private ListMultimap<String, LemPair> extractRandomLemPairs() {
		List<Integer> sizes = ImmutableList.of(50, 100, 200, 500, 1000, 5000, 10000, 50000);
		ListMultimap<String, LemPair> result = ArrayListMultimap.create();
		for (int size : sizes) {
			if (size > (lemVoc.size() * lemVoc.size())/2) {
				log.info(String.format("Not generating random lem2lem pairs for size %d since vocab is not big enough", size));
			}
			Set<LemPair> pairs = new HashSet<>();
			for (int i = 0; i < size; i++) {
				pairs.add(randomLemPair());
			}
			result.putAll(String.format("lem2lem_random_%d", size),  pairs); 
		}
		return result;
	}
	
	/*
	private ListMultimap<String, SynPair> extractRandomSynPairs() {
		List<Integer> sizes = ImmutableList.of(50, 100, 200, 500, 1000, 5000, 10000, 50000);
		ListMultimap<String, SynPair> result = ArrayListMultimap.create();
		for (int size : sizes) {
			if (size > (synVoc.size() * synVoc.size())/2) {
				log.info(String.format("Not generating random syn2syn pairs for size %d since vocab is not big enough", size));
			}
			Set<SynPair> pairs = new HashSet<>();
			for (int i = 0; i < size; i++) {
				pairs.add(randomSynPair());
			}
			result.putAll(String.format("syn2syn_random_%d", size),  pairs); 
		}
		return result;
	}
*/
	
	private LemPair randomLemPair() {
		return new LemPair(randomElement(lemVoc), randomElement(lemVoc));
	}
	
	/*
	private SynPair randomSynPair() {
		return new SynPair(randomElement(synVoc), randomElement(synVoc));
	}
	*/
	
	private <T> T randomElement(Set<T> set) {
		int index = rnd.nextInt(set.size());
		Iterator<T> iter = set.iterator();
		for (int i = 0; i < index; i++) {
		    iter.next();
		}
		return iter.next();
	}

//	private ListMultimap<String, SynDomPair> extractSynDomPairs(Set<Integer> synVoc, Set<String> domVoc) {
//		ListMultimap<String, SynDomPair> result = ArrayListMultimap.create();
//		int noDomCnt = 0;
//		for (int synId: synVoc) {
//			// does WordNet have domains (i.e. categories) for synsets?
//		}
//		log.info(String.format("Found %s syndom-pairs in vocab, %s syncons have no associated domain in sensigrafo.",
//				result.size(), noDomCnt));
//		return result;
//	}
	
	private Set<Synset> extractSynsets(List<String> vocab) {
		return vocab.stream()
				.filter(word -> isSynset(word))
				.map(word -> asSynset(word))
				.collect(Collectors.toSet());
	}
	
	private boolean isSynset(String word) {
		//first rule out various cases quickly
		if (word == null) return false;
		if (word.length() < 5) return false;
		if (word.charAt(4) != '_') return false;
		if (!word.startsWith("wn31_")) return false;
		Matcher matcher = synset_POS_Offset_Pattern.matcher(word);
		return matcher.matches();
	}
	
	private Synset asSynset(String word) {
		return decodeSynset_POS_Offset_SynsetSeqElt(word);
	}

	private Synset decodeSynset_POS_Offset_SynsetSeqElt(String dw) {
		Matcher matcher = synset_POS_Offset_Pattern.matcher(dw);
		if (!matcher.matches()) throw new IllegalArgumentException(String.format("Input '%s' is not a Synset elt", dw));
		try {
			String wnVersion = matcher.group(1);
			String wnPOSName = matcher.group(2);
			Long synsetOffset = Long.valueOf(matcher.group(3));
			return dict.getSynsetAt(POS.valueOf(wnPOSName), synsetOffset);
		} catch (IllegalStateException | JWNLException e) {
			throw new RuntimeException(String.format("Error decoding syncon for '%s'", dw), e);
		}
	}

	private Set<String> extractLemmas(List<String> vocab) {
		final String pre = "lem_";
		return vocab.stream()
			.filter(word -> word.startsWith(pre))
			.map(word -> word.substring(4))
			.filter(word -> isLemmaInWordNet(word))
			.map(word -> word)
			.collect(Collectors.toSet());
	}

	private boolean isLemmaInWordNet(String word) {
		IndexWordSet iws = getAllIndexWords(word);
		return iws.size() > 0;
	}

	static Options getCommandLineOptions() {
		Options options = new Options();

		options.addOption(Option.builder().longOpt(Opt.vocab.name()).argName("filename").hasArg()
				.desc("a vocab.txt file providing the seed WordNet vocab")
				.required().build());

		options.addOption(Option.builder().longOpt(Opt.out_dir.name()).argName("dirname").hasArg()
				.desc("path where the relation files will be written to. Default: <vocab_dir>/wn_vocabrels/")
				.build());
		
		return options;
	}
	
}
