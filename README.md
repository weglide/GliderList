![WeGlide gliderlist](./logo-gliderlist.png)
# gliderlist

Open and standardized collection of gliders. GliderList has two main goals:

* Provide a format suitable for exchanging glider types where no parsing is needed for both parties.
* Provide well structured and actively maintained data about gliders for different projects.

## Features

* Open and complete list of all gliders with corresponding index
* Strictly complying with naming convention of all manufactures
* Grouping of gliders with different engine options in groups having same structure
* Continuous integration of new index calculations every year in this repository (planned)
* Integration of different index lists (planned)

GliderList is used by WeGlide internally. WeGlide aims to provide numerical data on many aircraft in GliderList for people to do further research with.
GliderList will be actively maintained and compliance with naming conventions is and will be checked with manufacturers directly.

There exist two lists, base.csv and extended.csv. Base is a list of all base models (e.g. LS 8) where all the different models (e.g. LS 8T) are stated by regular expressions.
Extended is automatically generated out of base and includes all the different variants of the base models. Only extended includes the unique identifiers.

## Schema / Documentation

Intent of the following data schema is to be readable for humans and machines while enforcing strict rules and being open to extensions (e.g. other types of information for gliders like polars or numerical information).

### ID

Unique Identifier for this glider. ID is guaranteed not to change and can safely be used as identifier in databases.
New models will be assigned consecutive IDs.
  
### Glider

Name of the base model.

### Model

Name of the concrete configuration of the base model.
  
### Manufacturer

Manufacturer of the aircraft. Sometimes multiple manufacturers are possible so this is opinionated.

### Competition class
  
* 15
* 18
* Club
* Double
* Open
* Standard

### Winglets

Whether this aircraft has winglets in case the base model does not have winglets. This field is needed to calculate higher index (+1) based on base model.
**Warning**: This does not indicate if aircraft has winglets.

### Double Seater

Whether aircraft has two seats.

### FES

Whether engine is of kind FES (Front Electrical Sustainer). Only applicable if Engine evaluates to true.

### Engine

Whether aircraft has **any** kind of motorization.

### Year

Competition index (DMSt) for different years.

## Contribute

Contributions of new gliders or corrections of naming conventions are very welcome. Just open a pull request and we will review.
If adding a model in the extended list, leave the ID field empty as it is automatically added.
