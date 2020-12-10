package nn

// HiddenArrUint
type HiddenArrUint []uint

// HiddenLayer
func HiddenLayer(nums ...uint) HiddenArrUint {
	if len(nums) > 0 && nums[0] == 0 {
		return HiddenArrUint{}
	}
	return nums
}
