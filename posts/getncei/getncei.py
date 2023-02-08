import numpy as np
import requests
from scipy import spatial
from json.decoder import JSONDecodeError
from plotly.graph_objs import Scattermapbox, Layout, Figure
from math import radians, cos, sin, asin, sqrt
from itertools import islice
from typing import Any, Union, Optional

class CustomError(Exception):
    pass

class InputTypeErrorNCEI(CustomError):
    def __init__(self, message):
        super().__init__(message)

class InputValueErrorNCEI(CustomError):
    def __init__(self, message):
        super().__init__(message)    

class GetNCEI:
    """Initiation of API request to NCEI API Website.
    Get the data by requesting several endpoint of NCEI API url.
    
    Parameters\n
    ----------\n
    token (str): 
        Token to access web services, obtained from https://www.ncdc.noaa.gov/cdo-web/token.\n

    Methods\n
    -------\n
    get_datasets(filter, req_size): 
        Get the available datasets based on criteria of data specified by filter parameter, using "datasets" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/datasets).
    get_datacategories(filter, req_size): 
        Get the available datacategories based on criteria of data specified by filter parameter, using "datacategories" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/datacategories).
    get_datatypes(filter, req_size): 
        Get the available datatypes based on criteria of data specified by filter parameter, using "datatypes" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/datatypes).                                                
    get_locationcategories(filter, req_size): 
        Get the available locationcategories based on criteria of data specified by filter parameter, using "locationcategories" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/locationcategories).    
    get_locations(filter, req_size): 
        Get the available locations based on criteria of data specified by filter parameter, using "locations" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/locations).    
    get_stations(filter, req_size): 
        Get the available stations based on criteria of data specified by filter parameter, using "stations" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/stations).
    get_data(datasetid, filter, req_size): 
        Fetch the data from a (exactly one) dataset specified by datasetid parameter, based on criteria of data specified by filter parameter, using "data" endpoint (https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid={*datasetid}).                                            
    get_id_info(id, idcategory, idpairs): 
        Get the information of id parameter which is located in idcategory parameter. Can also be retrieved by specifying idpairs parameters as a dict of {id: idcategory}.    
    """

    def __init__(self, token: str) -> object:
        self._validate_input_type(token, [str])
        self._token= token

    def _validate_input_type(self, value, req_type):
        if type(value) not in req_type:
            raise (
                InputTypeErrorNCEI(
                    'Input Error, please check required input type. '
                    'See documentation.'
                    )
                )

    def _validate_input_value(
            self, value, min=None, max=None, equal=None
        ):
        raise_message = (
            'Input value Error, ' 
            'please check minimum, maximum, or required value. '
            'See documentation.'
            )
        if equal:
            if value != equal:
                raise (
                    InputValueErrorNCEI(raise_message)
                    )                
        elif min and max:
            if value < min or value > max:
                raise (
                    InputValueErrorNCEI(raise_message)
                    )

    def _validate_filter(
            self, filter, valid_filter
        ):
        self._validate_input_type(filter, [dict])
        raise_message = (
            'Input value Error, ' 
            'please check minimum, maximum, or required value. '
            'See documentation.'
            )
        for key in filter.keys():
            if key not in valid_filter:
                raise InputValueErrorNCEI(
                    raise_message
                )                  

    def _make_request(self, endpoint: str, payload=None):
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
        return (
            requests.get(
                url + endpoint, 
                headers={'token': self._token},
                params=payload
                )
            )

    def _iterpairs(
        self, count, req_size=None, req_limit=1000
        ):
        """
        Generate (limit, offset) pairs.
        """
        req_size = req_size or count
        if req_size < count:
            if req_size <= req_limit:
                yield (req_size, 1)
            else:
                req_done = 0
                for offset in range(1, req_size + 1, req_limit):
                    yield (min(req_limit, req_size - req_done), offset)
                    req_done += min(req_limit, req_size - req_done)
        else:
            for offset in range(1, count + 1, req_limit):
                yield(req_limit, offset)
                 
    def _store_response(self, endpoint, payload=None, req_size=None):
        req_limit = 1000
        if 'limit' in payload.keys() and payload['limit'] <= 1000:
            req_limit = payload['limit']
        temp_payload = payload.copy()
        if req_size:
            temp_payload['limit'] = min(req_limit, req_size)
        else:
            temp_payload['limit'] = req_limit
        try:
            response = \
                self._make_request(endpoint, temp_payload).json()
            count = response['metadata']['resultset']['count']
            iterpairs = self._iterpairs(count, req_size, req_limit)
            for limit, offset in islice(iterpairs, 1, None):
                temp_payload['offset'] = offset
                temp_payload['limit'] = limit
                response['results'] += \
                    self._make_request(endpoint, temp_payload)\
                        .json()['results']
        except KeyError:
            pass
        except JSONDecodeError:
            print(
                'Response not OK or data is not returned from API.'
                ' Response status: ',
                self._make_request(endpoint, temp_payload).status_code
                )
            raise
        return response
    
    def get_datasets(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {}, 
            req_size: Optional[int] = None
            ) -> list[dict[str, Union[str, int]]]:
        """Get the available datasets (using API request endpoint:'datasets'). All of the CDO data are in datasets. The containing dataset must be known before attempting to access its data.
        Criteria of the datasets available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the data sets that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datatypeid':
                VALUE (str or list[str]) -> Accepts a valid datatypeid or a list of datatypeids. Datasets returned will contain all of the data type(s) specified. Example: 'ACMH'.
            'locationid': 
                VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids.  Datasets returned will contain data for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
            'stationid': 
                VALUE (str or list[str]) -> Accepts a valid stationid or a list of stationids.  Datasets returned will contain data for the station(s) specified. Example: {'stationid': 'GHCND:ID000096745', ...}.                
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Datasets returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Datasets returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'stationid': ''GHCND:ID000096745'
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available datasets will be retrieved.
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain datatypes data, which contains fields of {'field1': 'values1', 'field2':'values2', ....}. The value associated within 'id' field can be used as 'datasetid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """    
        valid_filter = [
            'datatypeid', 'locationid', 'stationid', 
            'startdate', 'enddate',
            'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'datasets', filter, req_size
                )['results']
            )
        
    def get_datacategories(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {}, 
            req_size: Optional[int] = None
            ) -> list[dict[str, Union[str, int]]]:
        """Get the available datacategories (using API request endpoint:'datacategories'). Data Categories represent groupings of data types.
        Criteria of the datacategories available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the data categories that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datasetid':
                VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Data categories returned will be supported by dataset(s) specified. Example: 'GHCND'.
            'locationid': 
                VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids.  Data categories returned will be applicable for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
            'stationid': 
                VALUE (str or list[str]) -> Accepts a valid stationid or a list of stationids.  Data categories returned will be applicable for the station(s) specified. Example: {'stationid': 'GHCND:ID000096745', ...}.                
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data categories returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data categories returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'datasetid': 'GHCND',
                    'stationid': ''GHCND:ID000096745'
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available datacategories data will be retrieved.
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain datatypes data, which contains fields of {'field1': 'values1', 'field2':'values2', ....}. The value associated within 'id' field can be used as 'datacategoryid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datasetid', 'locationid', 'stationid',
            'startdate', 'enddate',
            'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'datacategories', filter, req_size
                )['results']
            )

    def get_datatypes(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {}, 
            req_size: Optional[int] = None
            ) -> list[dict[str, Union[str, int]]]:
        """Get the available datatypes (using API request endpoint:'datatypes'). Data Type describes the type of data, acts as a label. If it's 64°f out right now, then the data type is Air Temperature and the data is 64.
        Criteria of the datatypes available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the data types that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datasetid':
                VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Data types returned will be supported by dataset(s) specified. Example: 'GHCND'.
            'locationid': 
                VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids.  Data types returned will be applicable for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
            'stationid': 
                VALUE (str or list[str]) -> Accepts a valid stationid or a list of stationids.  Data types returned will be applicable for the station(s) specified. Example: {'stationid': 'GHCND:ID000096745', ...}.                
            'datacategoryid':
                VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids.  Data types returned will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP'}.
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data types returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data types returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'datasetid': 'GHCND',
                    'datacategoryid': 'TEMP',
                    'stationid': ''GHCND:ID000096745''
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available datatypes data will be retrieved.
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain datatypes data, which contains fields of {'field1': 'values1', 'field2':'values2', ....}. The value associated within 'id' field can be used as 'datatypeid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datasetid', 'locationid', 'stationid',
            'datacategoryid', 'startdate', 'enddate',
            'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'datatypes', filter, req_size
                )['results']
            )   

    def get_locationcategories(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {}, 
            req_size: Optional[int] = None
            ) -> list[dict[str, Union[str, int]]]:
        """Get the available locationcategories (using API request endpoint:'locationcategories'). Location categories are groupings of locations under an applicable label.
        Criteria of the locationcategories available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the location categories data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datasetid':
                VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Location categories returned will be supported by dataset(s) specified. Example: 'GHCND'.
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Location categories returned will have data after the specified date. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Location categories returned will have data before the specified date. Parameter can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'datasetid': 'GHCND',
                    'startdate': ''1970-10-03',
                    'sortfield': 'name'
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available locationcategories data will be retrieved.        
        
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain locationcategories data, which contains fields of {'field1': 'values1', 'field2':'values2', ...}. The value associated within 'id' field can be used as 'locationcategoryid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datasetid', 'startdate', 'enddate'
            'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'locationcategories', filter, req_size
                )['results']
            )

    def get_locations(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {},
            req_size: Optional[int] = None,
            ) -> list[dict[str, Union[str, int, float]]]:
        """Get the available locations (using API request endpoint:'locations'). Locations can be a specific latitude/longitude point such as a station, or a label representing a bounding area such as a city.
        Criteria of the locations available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the location data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datasetid':
                VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Locations returned will be supported by dataset(s) specified. Example: 'GHCND'.
            'locationcategoryid': 
                VALUE (str or list[str]) -> Accepts a valid locationcategoryid or a list of locationcategoryids. Locations returned will be in the location category(ies) specified. Example: {'locationcategoryid': 'CITY', ...}.
            'datacategoryid':
                VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids. Locations returned will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP', ...}.
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Locations returned will have data after the specified date. Parameter can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Locations returned will have data before the specified date. Parameter can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'datasetid': 'GHCND',
                    'locationcategoryid': 'CITY'
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available locations data will be retrieved.        
        
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain locations data, which contains fields of {'field1': 'values1', 'field2':'values2', ....}. The value associated within 'id' field can be used as 'locationid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datasetid', 'locationcategoryid', 'datacategoryid', 
            'startdate', 'enddate', 'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'locations', filter, req_size
                )['results']
            )

    def get_stations(
            self, 
            filter: Optional[dict[str, Union[str, list[str]]]] = {},
            req_size: Optional[int] = None
            ) -> list[dict[str, Union[str, int, float]]]:
        """Get the available stations (using API request endpoint:'stations'). Stations are where the data comes from (for most datasets) and can be considered the smallest granual of location data. If the desired station is known, all of its data can quickly be viewed
        Criteria of the stations available is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the station data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datasetid':
                VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Stations returned will be supported by dataset(s) specified. Example: 'GHCND'.
            'locationid': 
                VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids. Stations returned will contain data for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
            'datacategoryid':
                VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids. Stations returned will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP'}.
            'datatypeid': 
                VALUE (str or list[str]) -> Accepts a valid datatypeid or a list of datatypeids. Stations returned will contain all of the available data type(s) specified. Example: {'datatypeid': ['TAVG', 'TMAX', 'TMIN'], ...}.
            'extent':
                VALUE (str) -> The desired geographical extent for search. Designed to take a parameter generated by Google Maps API V3 LatLngBounds.toUrlValue. Stations returned must be located within the extent specified. Example: {'extent': '47.5204,-122.2047,47.6139,-122.1065', ...}
            'startdate':
                VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Stations returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
            'enddate':
                VALUE(str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Stations returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
            'sortfield': 
                VALUE (str = one from any of 'id', 'name', 'mindate', 'maxdate', 'datacoverage') -> Sort the results by the specified field. Example: {'sortfield': 'name', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                {
                    'datasetid': 'GHCND',
                    'datatypeid': ['TMAX', 'TMIN'],
                    'locationid': 'CITY:ID000008'
                    }

        req_size (int), Optional, default = None:
            Determining maximum row size of the data that will be retrieved. If not specified, all of the available stations data will be retrieved.
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain stations data, which contains fields of {'field1': 'values1', 'field2':'values2', ....}. The value associated within 'id' field can be used as 'stationid' as a filter for fetching the data using get_data() method or other get_* method.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datasetid', 'locationid', 'datacategoryid', 
            'datatypeid', 'extent', 'startdate', 'enddate'
            'sortfield', 'sortorder'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                )
        if req_size:
            self._validate_input_type(req_size, [int])
        return (
            self._store_response(
                'stations', filter, req_size
                )['results']
            )
    
    def get_data(
            self, 
            datasetid: str, 
            startdate: str,
            enddate: str,
            req_size: Optional[Union[int, str]] = 1000,
            filter: Optional[dict[str, Union[str, list[str]]]] = {}, 
            ) -> list[dict[str, Union[str, int, float,]]]:
        """Get the fetched data (using API request endpoint:'data') from a single datasetid. 
        Criteria of the data is specified by the filter parameter, and number of maximum rows returned is specified by req_size parameter.
        
        Parameters\n
        ----------\n
        datasetid (str), required:
            Datasetid of the data that want to be retrieved. Data returned will be from the datasetid specified. Example: 'GHCND'.
        startdate (str), required:
            Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data returned will be after the specified date. Annual and Monthly data will be limited to a ten year range while all other data will be limited to a one year range. Example: '1970-10-03'.
        enddate (str), required:
            Required. Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Data returned will be before the specified date. Annual and Monthly data will be limited to a ten year range while all other data will be limted to a one year range. Example: '2012-09-10'.
        req_size (int or str = 'all'), Optional, default = 1000:
            Determining maximum row size of the data that will be retrieved. req_size = 'all' will retrieve all of the available data.
        filter (dict[str, str | list[str]]), optional, default = {}: 
            Filter the data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
            KEYS:\n
            'datatypeid': 
                VALUE (str or list[str]) -> Accepts a valid datatypeid or a list of datatypeids. Data returned will contain all of the available data type(s) specified. Example: {'datatypeid': ['TAVG', 'TMAX', 'TMIN'], ...}.
            'locationid': 
                VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids. Data returned will contain data for the available location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
            'stationid': 
                VALUE (str or list[str]) -> Accepts a valid stationid or a list of stationids. Data returned will contain data for the available station(s) specified. Example: {'stationid': ['GHCND:ID000096745', 'GHCND:IDM00096739'], ...}.
            'units':
                VALUE (str = 'standard' or 'metric') -> Accepts the literal strings 'standard' or 'metric'. Data will be scaled and converted to the specified units. If a unit is not provided then no scaling nor conversion will take place. Example: {'unit': 'standard', ...).
            'sortfield': 
                VALUE (str = one from any of 'date', 'datatype', 'station', 'atribute', 'value') -> Sort the results by the specified field. Example: {'sortfield': 'value', ...}.
            'sortorder'
                VALUE (str = 'asc' or 'desc') -> Specifies whether sort is ascending or descending. Defaults to 'asc'. Example: {'sortorder': 'desc', ...}.
            Example:
                filter = {
                    'datatypeid': ['TMAX', 'TMIN'],
                    'stationid': 'GHCND:ID000096745'
                    }
        
        Returns\n
        -------\n
        list[dict]
            A list of dictionaries that contain data fields of {'field1': 'values1', 'field2':'values2', ....}.
        
        Raises\n
        ------\n
        InputTypeError
            If the input type of each parameters is not valid.
        InputValueError
            If the input value of req_size and filter keys are not valid.
        JSONDecodeError
            If there was an error with requesting API.
        """
        valid_filter = [
            'datatypeid', 'locationid', 'stationid', 
            'units', 'sortfield', 'sortorder'
            ]
        self._validate_input_type(datasetid, [str])
        self._validate_input_type(startdate, [str]) 
        self._validate_input_type(enddate, [str])      
        self._validate_input_type(req_size, [int, str]) 
        if type(req_size) == str:
            self._validate_input_value(value=req_size, equal='all')
            req_size = None
        if filter:  
            self._validate_filter(
                filter, valid_filter
                )
        payload = {
            'datasetid': datasetid,
            'startdate': startdate,
            'enddate': enddate
            }
        payload.update(filter)
        return (
            self._store_response(
                'data', payload, req_size
                )['results']
            )                

class FindDataIdNCEI(GetNCEI):
    """
    Find the datatypes that contains keyword, and inform in which datasets are they existed.

    Parameters
    ----------
    token (str): 
        Token to access web services, obtained from https://www.ncdc.noaa.gov/cdo-web/token.\n
    keyword (str or list[str]):
        Specify the keyword to search in various available datatypes.
    filter (dict[str, str | list[str]]), optional, default = {}: 
        Filter the datasets or datatypes that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
        KEYS:\n
        'locationid': 
            VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids. Matched datatypes that returned will be available for location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
        'stationid': 
            VALUE (str or list[str]) -> Accepts a valid stationid or a list of stationids. Matched datatypes that returned will be available for the station(s) specified. Example: {'stationid': ['GHCND:ID000096745', 'GHCND:IDM00096739'], ...}.
        'startdate':
            VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Matched datasets that returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
        'enddate':
            VALUE(str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Matched datasets that returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
        Example:
            filter = {
                'stationid': 'GHCND:ID000096745'
                }

    Methods\n
    -------_\n
    get_matched_datatypes(): Returns datatypes that its description contains any specified keyword.
    get_matched_datasets(): Returns datasets that contains matched datatypes.
    get_id_pairs(): Get id pairs of datasets-matched_datatypes stored in a dictionary. Those pairs contain datasetid and datatypeid that can be used for get_data() method or other get_*() method.
    """
    def __init__(
            self, 
            token: str, 
            keyword: Union[str, list[str]], 
            filter: dict = {}
            ):
        valid_filter = [
            'locationid', 'stationid', 
            'startdate', 'enddate'
            ]
        self._validate_input_type(token, [str])
        self._validate_input_type(keyword, [str, list])
        if type(keyword) == list:
            for key in keyword:
                self._validate_input_type(key, [str])
        if filter:
            self._validate_filter(
                filter, valid_filter
                )    
        self._token= token 
        self._temp_payload = filter.copy()
        if type(keyword) == list:
            self._keywords = keyword
        else:
            self._keywords = [keyword]
        self.matched_datatypes = []
        self.matched_datasets = []
        self.id_pairs = {}
        self._datasets = \
            self._store_response(
                'datasets', self._temp_payload
                )['results']
        for dataset in self._datasets:
            self._temp_payload['datasetid'] = dataset['id']
            temp_datatype = []
            try:
                datatypes = [
                    datatype for datatype in \
                        self._store_response(
                            'datatypes', self._temp_payload
                        )['results']
                    ]
                for datatype in datatypes:
                    if any(
                        keyword.lower() in datatype['name'].lower() 
                        for keyword in self._keywords
                        ):
                        self.matched_datatypes.append(datatype)
                        temp_datatype.append(datatype['id'])
                if temp_datatype:
                    self.id_pairs.update(
                        {dataset['id']: temp_datatype}
                        )
            except (KeyError, TypeError):
                continue
            if dataset['id'] in list(self.id_pairs.keys()):
                self.matched_datasets.append(dataset)

    def get_matched_datasets(self) -> list[dict[str, Union[str, int]]]:
        """Returns datasets that have any datatypes that contains specified keyword.

        Parameter\n
        ---------\n
        None

        Returns\n
        -------\n
        list[dict]:
            List of datasets that have any datatypes for specified keyword
        """
        return self.matched_datasets

    def get_matched_datatypes(self) -> list[dict[str, Union[str, int]]]:
        """Returns datatypes that contains specified keyword.

        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        list[dict]:
            List of datatypes that contains specified keyword
        """
        return self.matched_datatypes

    
    def get_id_pairs(self) -> dict[str, list[str]]:
        """Returns pairs of datasets-matched datatypes.

        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        dict[str: list[str]]:
            Dictionary of {matched_datasetid1: [matched_datatypeids], matched_datasetid2: [matched_datatypeids], ...}. Can be used as datasetid and datatypeid for get_data() method or other get_* method().
        """
        return self.id_pairs

class FindLocationInfoNCEI(GetNCEI):
    """Find a location of available data by searching it based on target keyword. Matched location can be filtered using filter parameter to verify that location will contains that specified features.

    Parameters\n
    ----------\n
    token (str): 
        Token to access web services, obtained from https://www.ncdc.noaa.gov/cdo-web/token.\n
    target (str):
        Specify the keyword to search in various available locations. Example: 'New York'.
    locationcategoryid (str):
        As a category which describes the scope of target keyword. Example: 'CITY', as a suited value if 'New York' was specified in target parameter. 
    filter (dict[str, str | list[str]]), optional, default = {}: 
        Filter the datasets or datatypes that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
        KEYS:\n
        'datasetid':
            VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Locations returned will match with keyword and will be supported by dataset(s) specified. Example: {'datasetid': 'GHCND', ...}.
        'datacategoryid':
            VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids. Locations returned will match with keyword and will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP', ...}.
        'startdate':
            VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Locations returned will match with keyword and will have data after the specified date. Parameter can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
        'enddate':
            VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Locations returned will match with keyword and will have data before the specified date. Parameter can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
        Example:
            filter = {
                'datasetid': 'GHCND',
                'datacategoryid': 'TEMP'
                }    

    Methods\n
    -------\n
    get_location_info: Returns the location info in a dictionary that matched the exact target keyword.
    get_locationid: Returns the id of matched location.
    """
    def __init__(
            self, token: str, 
            target: str, 
            locationcategoryid: str, 
            filter: Optional[dict] = {}
            ) -> object:
        self._validate_input_type(token, [str])
        self._validate_input_type(target, [str])
        self._validate_input_type(locationcategoryid, [str])        
        valid_filter = [
            'datasetid', 'datacategoryid', 
            'startdate', 'enddate'
            ]
        if filter:  
            self._validate_filter(
                filter, valid_filter
                )              
        self._target = target
        self._token = token
        self._endpoint = 'locations'
        self._payload = {
            'sortfield': 'name',
            'locationcategoryid': locationcategoryid.upper(),
            'limit': 1
            }
        self._payload.update(filter)
        self._key = 'name'
        self._info = {}
        self._nresponses = 0

    def _get_item(
            self, target, key, payload, endpoint, start=1, end=None
            ):
        try:
            target = target.lower()
        except AttributeError:
            pass
        mid = (start + (end or 1)) // 2
        temp_payload = payload.copy() 
        temp_payload['offset'] = mid
        response = self._make_request(endpoint, temp_payload)
        if response.ok:
            count = \
                response.json()['metadata']['resultset']['count']
            end = end or count
            try:
                current_value = \
                    response.json()['results'][0][key].lower()
            except AttributeError:
                current_value = \
                    response.json()['results'][0][key]
            if target in current_value:
                return response.json()['results'][0]
            else:
                if start >= end:
                    return {}
                elif target < current_value:
                    return \
                        self._get_item(
                            target, key, payload, endpoint, 
                            start, mid - 1
                            )
                elif target > current_value:
                    return \
                        self._get_item(
                            target, key, payload, endpoint, 
                            mid + 1, end
                            )
        else:
            print(
                f'Response not OK, status: {response.status_code}'
                )
        
    def get_location_info(self) -> dict[str, Union[str, int, float]]:
        """
        Get the location info as a dict that match the target keyword.
        
        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        dict:
            Dictionary that contains location info that matched with target keyword.
        """
        self._info = self._get_item(
            self._target, self._key, self._payload, self._endpoint
            )
        return self._info

class FindStationInfoNCEI(GetNCEI):
    """
    Find available stations that contains specified target keyword.

    Parameters\n
    ----------\n
    token (str): 
        Token to access web services, obtained from https://www.ncdc.noaa.gov/cdo-web/token.\n
    target (str):
        Specify the keyword to search in various available stations. Example: 'Salt Lake' to find stations that contains this keyword in its description.
    filter (dict):
        Filter the station data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below:\n
        KEYS:\n
        'datasetid':
            VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Matched stations returned will be supported by dataset(s) specified. Example: {'datasetid': 'GHCND'}.
        'locationid': 
            VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids. Matched stations returned will contain data for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
        'datacategoryid':
            VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids. Matched stationss returned will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP'}.
        'datatypeid': 
            VALUE (str or list[str]) -> Accepts a valid datatypeid or a list of datatypeids. Matched stations returned will contain all of the available data type(s) specified. Example: {'datatypeid': ['TAVG', 'TMAX', 'TMIN'], ...}.
        'extent':
            VALUE (str) -> The desired geographical extent for search. Designed to take a parameter generated by Google Maps API V3 LatLngBounds.toUrlValue. Stations returned must be located within the extent specified. Example: {'extent': '47.5204,-122.2047,47.6139,-122.1065', ...}
        'startdate':
            VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Matched stations returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
        'enddate':
            VALUE(str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Matched stations returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
    
    Methods\n
    -------\n
    get_station_info: Returns all stations that contain target keyword in its description.
    """
    def __init__(
            self, token: str, 
            target: Union[str, list[str]], 
            filter: Optional[dict] = {}
            ) -> object:
        self._validate_input_type(token, [str])
        self._validate_input_type(target, [str, list])
        if type(target) == list:
            for _target in target:
                self._validate_input_type(_target, [str])
        valid_filter = [
            'datasetid', 'locationid', 'datacategoryid', 
            'datatypeid', 'extent', 'startdate', 'enddate'
            ]
        if filter:  
            self._validate_filter(
                filter, valid_filter
                )         
        if type(target) == list:
            self._target = target.copy()
        else:
            self._target = [target]
        self._token = token
        self._payload = {
            'sortfield': 'name',
            'limit': 1000
            }
        self._payload.update(filter)
        self._matched_stations = []
        self._nresponses = 0

    def _multi_requests(self, count, payload):
        temp_payload = payload.copy()
        limit = temp_payload['limit']
        for offset in range(limit + 1, count + 1, limit):
            temp_payload['offset'] = offset
            self._nresponses += 1
            yield (
                self._make_request('stations', temp_payload)\
                    .json()['results']
                ) 

    def _search_arr(self, stations):
        stations_arr = np.array(stations)
        stations_name_arr = \
            np.array([station['name'].lower() for station in stations])
        match_indices = np.empty(0, dtype=np.int64)
        for keyword in self._target:
            match_indices = \
                np.concatenate(
                    (match_indices,
                     np.where(
                            np.char.find(
                                stations_name_arr, keyword.lower()
                                ) != -1
                            )), axis=None
                )   
        self._matched_stations \
            += list(stations_arr[np.unique(match_indices)])
             
    def _get_matched_stations(self):
        response = self._make_request('stations', self._payload)
        self._nresponses += 1
        try:
            count = response.json()['metadata']['resultset']['count']
        except JSONDecodeError:
            print(
                'Response not OK, status:',
                response.status_code
                )
            raise               
        limit = self._payload['limit']
        stations = response.json()['results']
        self._search_arr(stations)
        if count > limit:
            for _stations in (
                    self._multi_requests(count, self._payload)
                    ):
                self._search_arr(_stations)

    def get_station_info(self) -> list[dict[str, Union[str, int, float]]]:
        """"Returns all stations that contain target keyword in its description.
        
        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        list[dict]:
            List of dictionaries of matched stations.
        """
        self._get_matched_stations()
        return self._matched_stations
    
class FindNearestStationNCEI(FindStationInfoNCEI):
    """ Find the nearest station with the specified coordinate.

    Parameters\n
    ----------\n
    token (str): 
        Token to access web services, obtained from https://www.ncdc.noaa.gov/cdo-web/token.\n
    coord (tuple):
        Tuple of (lat, long) decimal degree coordinate. The latitude (decimated degrees w/northern hemisphere values > 0, southern hemisphere values < 0), longitude (decimated degrees w/western hemisphere values < 0, eastern hemisphere values > 0).
    filter (dict), optional, default = {}:
        Filter the station data that will be retrieved using a Dict, which the KEYS are the 'Additional Parameters' for the API request. Accepted {KEYS:VALUES} pairs are as explained below: \n
        KEYS: \n
        'datasetid':
            VALUE (str or list[str]) -> Accepts a valid datasetid or a list of datasetids. Nearest stations returned will be supported by dataset(s) specified. Example: {'datasetid': 'GHCND'}.
        'locationid': 
            VALUE (str or list[str]) -> Accepts a valid locationid or a list of locationids. Nearest stations returned will contain data for the location(s) specified. Example: {'locationid': ['FIPS:37', 'CITY:ID000008'], ...}.
        'datacategoryid':
            VALUE (str or list[str]) -> Accepts a valid datacategoryid or a list of datacategoryids. Nearest stationss returned will be associated with the data category(ies) specified. Example: {'datacategoryid': 'TEMP'}.
        'datatypeid': 
            VALUE (str or list[str]) -> Accepts a valid datatypeid or a list of datatypeids. Nearest stations returned will contain all of the available data type(s) specified. Example: {'datatypeid': ['TAVG', 'TMAX', 'TMIN'], ...}.
        'startdate':
            VALUE (str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Nearest stations returned will have data after the specified date. Paramater can be use independently of 'enddate'. Example: {'startdate': '1970-10-03', ...}.
        'enddate':
            VALUE(str) -> Accepts a valid ISO formated date (YYYY-MM-DD) or date time (YYYY-MM-DDThh:mm:ss). Nearest stations returned will have data before the specified date. Paramater can be use independently of 'startdate'. Example: {'enddate': '2012-09-10', ...}.
    station_nos (int), optional, default = 1:
        Number of nearest stations that wanted to be returned.
    Methods\n
    -------\n
    get_nearest_station(): Return a station info that located nearest with specified target coordinate.
    show_location(): Show a cartographic plot of target coordinate compared with nearest station found.  
    """
    def __init__(
            self, token, 
            coord: tuple[Union[int, float], Union[int, float]],
            filter: Optional[dict] = {},
            station_nos: Optional[int] = 1 
            ) -> object:
        self._validate_input_type(token, [str])
        self._validate_input_type(coord, [tuple])
        self._validate_input_type(station_nos, [int])
        self._station_nos = station_nos
        for coord_value in coord:
            self._validate_input_type(coord_value, [int, float])
        valid_filter = [
            'datasetid', 'locationid', 'datacategoryid', 
            'datatypeid','startdate', 'enddate'
            ]
        if filter:
            self._validate_filter(
                filter, valid_filter
                ) 
        self._token = token
        self._target_coord = coord
        self._payload = {
            'sortfield': 'name',
            'limit': 1000
            }
        self._payload.update(filter) 
        self._result = {}
        self._distance = None
        self._nresponses = 0
    
    def _coord_distance(self, point_1, point_2):
        """
        """
        lat1, lon1 = map(radians, [point_1[0], point_1[1]])
        lat2, lon2 = map(radians, [point_2[0], point_2[1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        # Haversine
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))   
        r = 6371 #km
        return (c * r)

    def _query_nearest(self, stations):
        stations_coord = [
            (station['latitude'], station['longitude'])
                for station in stations
            ]
        query = \
            spatial.KDTree(stations_coord)\
                   .query(self._target_coord, self._station_nos)
        return (
            query[0], 
            [stations[index] for index in query[1]]
            )

    def _update_nearest(
            self, 
            distance1, stations1,
            distance2, stations2
            ):
        pair1 = np.array([distance1, stations1])
        pair2 = np.array([distance2, stations2])
        updated_pair = \
            np.where(pair2[0] < pair1[0], pair2, pair1)
        self._distance = updated_pair[0]
        self._result = updated_pair[1]
    
    def _nearest_station(self):
        response = self._make_request('stations', self._payload)        
        try:
            count = response.json()['metadata']['resultset']['count']
        except JSONDecodeError:
            print(
                'Response not OK, status:',
                response.status_code
                )
            raise               
        limit = self._payload['limit']
        self._distance, self._result = \
            self._query_nearest(response.json()['results'])
        if count > limit:
            iterstations = self._multi_requests(count, self._payload)
            for stations in iterstations:
                temp_distance, temp_station = \
                    self._query_nearest(stations)
                self._update_nearest(
                    self._distance, self._result,
                    temp_distance, temp_station
                )
                
    
    def get_nearest_station(self) -> dict[str, Union[str, int, float]]:
        """Return a station info that placed nearest with specified target coordinate.
        
        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        dict:
            Nearest station info fields stored as a dictionary.
        """
        if not self._result:
            self._nearest_station()
        return self._result

    def show_location(self) -> object:
        """
        Plotting nearest station and target coordinate on a cartographic map.
        
        Parameters\n
        ----------\n
        None

        Returns\n
        -------\n
        Plotly.Figure object:
            Nearest station plot that located nearest to the target coordinate.
        """
        if not self._result:
            self._nearest_station()
        target_lat, target_lon = self._target_coord   
        target_text = 'Specified Target'
        plot_data = [
            Scattermapbox(
                lat=[target_lat],
                lon=[target_lon],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=15 
                    ),
                text=target_text,
                name='Target'
                )
            ]
        for station in self._result:
            station_lat, station_lon = (
                station['latitude'],
                station['longitude']
                )            
            distance = self._coord_distance(
                (target_lat, target_lon),
                (station_lat, station_lon)
                )     
            result_text = (
                station['name'] 
                + f'.\nDistance approx.: {distance:.2f} km'
                )
            plot_data.append(
                Scattermapbox(
                    lat=[station_lat],
                    lon=[station_lon],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15
                        ),
                    text=result_text,
                    name='Nearest Station',
                    )
            )
        plot_layout = Layout(title='Nearest Station')
        fig = Figure(data=plot_data, layout=plot_layout)
        fig.update_layout(
            mapbox_style="open-street-map",            
            autosize=True,
            hovermode='closest',
            mapbox=dict(
                bearing=0,
                center=dict(
                    lat=target_lat,
                    lon=target_lon
                    ),
                pitch=0,
                zoom=10
                ),
            )
        fig.show(renderer='notebook')