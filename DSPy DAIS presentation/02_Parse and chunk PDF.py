# Databricks notebook source
# MAGIC %md
# MAGIC # Parsing and Chunking Summary of benefits

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read PDF documents

# COMMAND ----------

# MAGIC %md
# MAGIC #####Import utility methods

# COMMAND ----------

# MAGIC %pip install docling
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/init

# COMMAND ----------

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

document_name = ["SBC_client1.pdf", "SBC_client2.pdf", "SBC_client3.pdf", "SBC_client4.pdf"]
x=0
all_chunks = []
while x < len(document_name):
  conv_res = DocumentConverter().convert(f"/Volumes/austin_choi_demo_catalog/demo_data/sbc/{document_name[x]}")
  doc = conv_res.document
  chunker = HybridChunker()
  chunk_iter = chunker.chunk(dl_doc=doc)

  for i, chunk in enumerate(chunk_iter):
    enriched_text = chunker.serialize(chunk=chunk)

    all_chunks.append({
      "document_name": document_name[x],
      "chunk_id": f"{document_name[x]}_Chunk_{i}",
      "chunk_index": i,
      "content": enriched_text
    })
  x+=1
all_chunks

# COMMAND ----------

chunks_df = spark.createDataFrame(all_chunks)

# COMMAND ----------

display(chunks_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Save the SBC data to a Delta table in Unity Catalog

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{sbc_details_table_name}")
chunks_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{sbc_details_table_name}")

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.{sbc_details_table_name}"))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `austin_choi_demo_catalog`.`demo_data`.`sbc_details` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------


