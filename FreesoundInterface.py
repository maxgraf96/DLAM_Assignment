import requests

"""
Workflow of this process: Instantiate class and pass AUTH_CODE
"""
class FreesoundInterface():
    def __init__(self, auth_code):
        self.client_id = "bLrLdiV8lf8Pk8D8oEDC"
        self.api_key = "RbBBCqZxsPym6J0ykeFpawC0HWuPRebVcgtSv2uU"
        # Update this on new days :)
        self.auth_code = auth_code
        self.license = "\"Creative Commons 0\""
        self.sample_rate = "44100"
        self.duration_from = "3.0"
        self.duration_to = "5.0"

        # Establish connection
        self.establish_connection()

    def establish_connection(self):
        # Check if access token exists and create a new one if there is not currently one saved
        # NB: Doesn't check if token is expired atm
        if len(self.get_access_token_from_file()) == 0:
            access_token = self.get_access_token()
            self.save_access_token_to_file(access_token)
            print("New access token created.")

    def search(self, term, tags, ac_single_event=True, ac_note_name=None):
        """
        Search for a given term.
        :param term: The search term
        :param tags: The tags that should be associated with the sound
        :param ac_note_name: The name of the note to be queried:
        Must be one of [“A”, “A#”, “B”, “C”, “C#”, “D”, “D#”, “E”, “F”, “F#”, “G”, “G#”]
        and the octave number. E.g. “A4”, “E#7”.
        :return: List of results
        """
        response = self.create_search_request(term, tags, ac_single_event, ac_note_name)
        results_list = response["results"]

        return results_list

    def authorise(self):
        QUERY = "https://freesound.org/apiv2/oauth2/authorize/"
        headers = {'client_id': self.client_id, 'response_type': 'code'}
        response = requests.get(QUERY, headers=headers)
        return response

    def get_access_token(self):
        QUERY = "https://freesound.org/apiv2/oauth2/access_token/"
        payload = {
            'client_id': self.client_id,
            'grant_type': 'authorization_code',
            'client_secret': self.api_key,
            'code': self.auth_code
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(QUERY, data=payload, headers=headers)
        return response.json()["access_token"]

    def create_search_request(self, term, tags, ac_single_event, ac_note_name):
        """
        Creates and makes an API search call to freesound.org with the specified parameters
        :param term: The search term
        :param tags: The tags associated with the sound
        :param ac_single_event: Whether only sounds containing a single sound event should be queried
        :param ac_note_name:
        :return: The API response (in Python dict format)
        """
        # Check term
        if term is None:
            term = ""
        # Convert tags from list to string
        if tags is not None and len(tags) > 0:
            tags_str = ""
            for tag in tags:
                tags_str = tags_str + " tag:" + tag
        else:
            tags_str = ""

        # Create string for ac_single_event
        ac_single_event_str = " ac_single_event: True" if ac_single_event else ""

        # Create note name string
        if ac_note_name is not None:
            ac_note_name_str = " ac_note_name: " + ac_note_name + ""
            # ac_note_name_str = ""
            # for i in range(8):
            #     ac_note_name_str = ac_note_name_str + " ac_note_name: " + ac_note_name + str(i)
        else:
            ac_note_name_str = ""

        FILTERS = "&filter=license:" + self.license + " " \
                  + "duration:[" + self.duration_from + " TO " + self.duration_to + "]" \
                  + tags_str + ac_single_event_str + ac_note_name_str
        FILTERS = FILTERS + " " + "samplerate:" + self.sample_rate + " " + "type:wav"
        QUERY = "https://freesound.org/apiv2/search/text/?query=" + term + FILTERS + "&token=" + self.api_key
        response = requests.get(QUERY)

        if response.status_code == 200:
            number_of_results = response.json()["count"]
            print("Searching for '" + term + "' with tags '" + ", ".join(tags) + "' and note " + ac_note_name_str + ". Got " + str(number_of_results) + " results.")
        else:
            print("Error while searching for " + term + ". Status code " + str(response.status_code))
        return response.json()

    def download_pack(self, pack_id):
        QUERY = "https://freesound.org/apiv2/packs/" + pack_id + "/download/"
        access_token = self.get_access_token_from_file()
        headers = {'Authorization': 'Bearer ' + access_token + ''}
        response = requests.get(QUERY, headers=headers)
        return response.content

    def download_sound(self, id):
        QUERY = "https://freesound.org/apiv2/sounds/" + id + "/download/"
        access_token = self.get_access_token_from_file()
        headers = {'Authorization': 'Bearer ' + access_token + ''}
        response = requests.get(QUERY, headers=headers)
        return response.content

    def save_access_token_to_file(self, token):
        text_file = open("access_token.txt", "w")
        text_file.write(token)
        text_file.close()

    def get_access_token_from_file(self):
        return open("access_token.txt", "r").read()