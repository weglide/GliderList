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

## Schema / Documentation

Intent of the following data schema is to be readable for humans and machines while enforcing strict rules and being open to extensions (e.g. other types of information for gliders like polars or numerical information).

## What models are included

Gliders should have a unique row in gliderlist if their handling or engine option differs from existing types.

A unique row is applicable if the aircraft has (compared to existing type)

* Increased MTOW
* Winglets
* Different fuselage / wings

It is not applicable if the aircraft has (compared to existing type)

* Automatic control connections
* A different cockpit
* A different main wheel / break-system

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

### Kind

* GL -> Gl
* MG -> Motorglider (Engine)
* FG -> FES Glider (FES)

### Double Seater

Whether aircraft has two seats.

### Winglets

Whether this aircraft has winglets in case the base model does not have winglets. This field is needed to calculate higher index (+1) based on base model.
**Warning**: This does not indicate if aircraft has winglets.

### Exclude from Live

Indication if most flights with this aircraft are done in pure powered mode.

### Year

Competition index (DMSt) for different years.

## Contribute

Contributions of new gliders or corrections of naming conventions are very welcome. Just open a pull request and we will review.
Please consider whether the model applies to be added.
