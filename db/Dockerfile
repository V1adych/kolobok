FROM mysql:8.0

COPY sizes.sql            /docker-entrypoint-initdb.d/
COPY marks_models.sql     /docker-entrypoint-initdb.d/

VOLUME /var/lib/mysql
EXPOSE 3306