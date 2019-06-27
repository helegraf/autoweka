package experiments;

import org.aeonbits.owner.Config.Key;

import experiments.two_part.part_two.execution.RankerConfig;

public interface AutoWEKARankerConfig extends RankerConfig {

	@Key("autoweka.seed")
	public int getSeed();
	
	@Key("autoweka.numCPUs")
	public int getNumCPUs();
	
	@Key("autoweka.memory")
	public int getMemory();
	
	@Key("autoweka.totalTimeoutSeconds")
	public int getTotalTimeoutSeconds();
	
	@Key("autoweka.evaluationTimeoutSeconds")
	public int getEvaluationTimeoutSeconds();
	
	@Key("autoweka.searchSpace")
	public String getSearchSpace();
	
	@Key("db.upload_intermediate_results")
	public boolean uploadIntermediateResults();
	
	@Key("db.host")
	public String getHost();
	
	@Key("db.user")
	public String getUser();
	
	@Key("db.pw")
	public String getPassword();
	
	@Key("db.db")
	public String getDatabase();
	
	@Key("db.intermediate_results_table")
	public String getIntermediateResultsTable();
}
