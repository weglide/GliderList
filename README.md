# GliderList

Open and standardized glider directory.

## Schema

Intent of the following data schema is to be readable for humans and machines while enforcing strict rules and beeing open to extensions (e.g. other types of air sport devices).

## Documentation

This is the Definition of the list attributes in the following form:

* **Attribute Name**: Type

 _Description._
  
 1. Choice 1 (if restricted)

If a value is not known, it should be displayed as ``-``.

### Attributes

* **short name**: Custom hash function

 _Human decodable unique identifier._

* **type**: Number

 _Type of aircraft._
  
 1. glider
 2. hang glider
 3. paraglider
 4. model glider
  
* **manufacturer**: String

 _Manufacturer of the aircraft._

* **model**: String

 _Aircraft model (without variants)._

* **variant**: String

 _Variant of the aircraft model. Create one aircraft for every variant._
  
* **class**: Number

 _Competition class._
  
 1. open
 2. 18
 3. 15
 4. standard
 5. club
 6. double
 
 * **motor**: Boolean
 
 _Has any kind of motor._
 
* **competition-year** (e.g. dmst-2020): Number

 _Competition index for different competitions and years._ 

## Contribute

Create pull requests here on GitHub. 
