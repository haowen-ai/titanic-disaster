FROM rocker/r-ver:4.6.2

WORKDIR /app

COPY src/r/install_packages.R src/r/install_packages.R
RUN Rscript src/r/install_packages.R

COPY src/r/ src/r/
COPY src/data/ src/data/

CMD ["Rscript", "src/r/titanic.R"]
