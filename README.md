# Query Auto-Completion for Rare Prefixes
The log files should be included in the `data/logs` directory.
[Download AOL Query Log](http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/)

## AOL Query Log 
The AOL query Log consists of 5 components, with each at least containing an user id (AnonID), the actual query (Query), and timestamp (QueryTime). Additional, the rank (ItemRank) of the record clicked and the URL (ClickURL) of the clicked record are available if applicable. An empty Duplicate queries at the same time with a non-empty ItemRank and ClickURL means that the user has clicked on multiple links for the searched query.

| AnonID      | Query                            | QueryTime                 | ItemRank      | ClickURL                        |
|-------------|----------------------------------|---------------------------|---------------|---------------------------------|
| 1337	      | select business services	     | 2006-03-14 15:51:41	     |               |                                 |
| 1337	      | select business services title	 | 2006-03-14 15:52:10	     |               |                                 |
| 1337	      | cbc companies	                 | 2006-03-14 15:52:44	     | 2             | http://www.cbc-companies.com    |
| 1337	      | cbc companies	                 | 2006-03-14 15:52:44	     | 3             | http://www.cbc-companies.com    |
| 1337	      | cbc companies	     			 | 2006-03-14 15:52:44		 | 4             | http://www.mktgservices.com     |


## Requirements
| Name        | Version | Install                   | Description                      |
|-------------|---------|---------------------------|----------------------------------|
| numpy       | 1.18.1  | `pip install numpy`       | Used for scientific computing.   |

	
