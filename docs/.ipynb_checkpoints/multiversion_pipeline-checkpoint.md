
#### Steps to enable concurrent operation of multiple system versions:
1. copy over relevant cached models to model_cache_dir (e.g. albert-base-v2-pytorch_model.bin, albert-base-v2-spiece.model)
    ```bash mkdir ~/datasets/model_cache/<branch_name>
    cp ~/datasets/model_cache/albert-base-v2* ~/datasets/model_cache/<branch_name>/
    ```
1. seed tweets
    ```sql
    insert into deep_classiflie_dev.dcbot_tweets (thread_id, end_thread_tweet_id, statement_text, t_start_date, t_end_date, retweet, statement_hash) select thread_id, end_thread_tweet_id, statement_text, t_start_date, t_end_date, retweet, statement_hash from deep_classiflie.dcbot_tweets;
    ```
1. if you want to avoid waiting for manual reload of wp_statement and fbase_statement archives, manually seed
    ```sql
    INSERT INTO deep_classiflie_dev.wp_statements (iid, statement_text, repeats, topic, source, pinnochios, s_date, statement_hash) select iid, statement_text, repeats, topic, source, pinnochios, s_date, statement_hash from deep_classiflie.wp_statements;
    INSERT INTO deep_classiflie_dev.fbase_transcripts select * from deep_classiflie.fbase_transcripts;
    INSERT INTO deep_classiflie_dev.fbase_statements (tid, sid, statement_text, sentiment) SELECT tid, sid, statement_text, sentiment FROM deep_classiflie.fbase_statements;
    ```
1. seed tweetbot credentials
    ```sql
    insert into deep_classiflie_dev.dcbot_creds select * from deep_classiflie.dcbot_creds;
     ```
 1. seed tweetbot reporting tables
    ```sql
    insert into tweets_analyzed_notpublished select thread_id, 'bot initialized, subsequent tweets will be processed', t_end_date from dcbot_tweets where 
    thread_id=(select max(thread_id) from dcbot_tweets)
    insert into stmts_analyzed_notpublished select s.tid, s.sid, 'bot initialized, subsequent stmts will be processed', t.t_date from fbase_statements s, 
    fbase_transcripts t where t.tid=s.tid and t.t_date=(select max(t_date) from fbase_transcripts)
    ```

1. copy and config environmental config files for both the modeling and data source projects into user HOME using examples (files will usually be the same but data source project env needed if running it independently)
    ```bash
    cp db_setup/.some_db_feature_branch_config.example ~/
    cp utils/.some_feature_branch_config.example ~/
    ```
