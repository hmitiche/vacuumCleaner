#kezrane margueritte 
# tp ai 
	"""	
	greedy agent for 'vacuum-3rooms-v2' world, 
	check the map in 'maps.py'
	author: kezrane noor
	"""	
	def agent_3rooms_v1(self, location, dirty):
		# ida kant dirty
		if dirty: 	
			action = self._action_dict['suck']
		else:
			# dirt mayrj3sh w deja visit all rooms
			if self._eco_mode and not self._dirt_comeback:
				if self._visited == 3:
					return self._action_dict['none'] #mandir walo
			# fel 0 w clean
			if np.array_equal(location, self._locations[0]):
				action = self._action_dict['down']
			# fel 1 w clean 
			elif np.array_equal(location, self._locations[1]):	
				action = self._action_dict['right']
			else:	#fel nos
				# ida kant fel nos w jaya mel 1
				if np.array_equal(self._last_location, self._locations[1]):
					action = self._action_dict['up']
				else:	#ida kant fel nos w jaya mel 0
					action = self._action_dict['left']
		return action