#Simple k-means classifer run on a shapfile input
library(sp)
library(raster)
library(classInt)
library(rgdal)
library(dplyr)

# Set workspace
setwd("C:/your/workspace/here")

# Set input shapefile
shapefile <- "your_shapefile.shp"

# Generate a data frame with 50 rows of randomly generated ID and value fields
data_frame <- data.frame(ID = paste0("ID", 1:50), Value = rnorm(50))

# Read in the shapefile as a SpatialPolygonsDataFrame object
shapefile_spdf <- readOGR(dsn = ".", layer = shapefile)

# Join the data frame to the shapefile based on a shared ID using the left_join function
shapefile_spdf <- left_join(shapefile_spdf, data_frame, by = "ID")

# Conduct a K-means classification on the Value field in the shapefile using the classIntervals function
kmeans_classes <- classIntervals(shapefile_spdf$Value, n = 3, style = "kmeans")

# Add the output classification as a new variable to the shapefile
shapefile_spdf$Class <- cut(shapefile_spdf$Value, kmeans_classes$brks, labels = kmeans_classes$brks[-1], include.lowest = TRUE)

# Write the updated shapefile to a new file
writeOGR(shapefile_spdf, ".", "output_shapefile", driver = "ESRI Shapefile")
