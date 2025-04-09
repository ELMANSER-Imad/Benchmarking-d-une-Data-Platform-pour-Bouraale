## Base image
#FROM bitnami/spark:latest

# Install Iceberg dependencies
#RUN mkdir -p /opt/spark/jars
#COPY iceberg-spark-runtime-*.jar /opt/spark/jars/

# Set Spark configurations for Iceberg
#RUN echo "spark.sql.catalog.demo=org.apache.iceberg.spark.SparkCatalog" >> /opt/spark/conf/spark-defaults.conf && \
#    echo "spark.sql.catalog.demo.type=hadoop" >> /opt/spark/conf/spark-defaults.conf && \
#    echo "spark.sql.catalog.demo.warehouse=/tmp/warehouse" >> /opt/spark/conf/spark-defaults.conf

# Expose Spark ports
#EXPOSE 8080 7077

# Default command to start Spark Master
#CMD ["/opt/bitnami/spark/bin/spark-class", "org.apache.spark.deploy.master.Master"]