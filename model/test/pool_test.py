
# Define the target and population matrices
ind = np.zeros((7, 15, 15))
ind[1, :, 3:5] = 1
ind[3, :, -2:] = 1
ind[-1, -1, :5] = [2, 1, 50, 5, .1]
ind[-1, 0, :3] = [.9, .1, 6]
ind[-1, 2, :3] = [.9, .1, 8]
ind[-2, 0, :2] = [0, 2]
ind[-2, 1, :4] = [.6, .1, .1, 4]

t = np.full((30, 30), fill_value=12)
population = [ind, ind, ind]

# Create an instance of PoolingLayers
obj = PoolingLayers(
    target=t,
    pooling_method="max",
    pool_size=(2, 2),
    strides=(2, 2),
    padding="valid",
    zero_padding=(2, 2),
    kernel_size=(2, 2),
    up_padding="valid",
    up_strides=(2, 2)
)

# Test down_sample_target method
down_sampled = obj.down_sample_target(
    target=t,
    pooling_method=obj.pooling_method,
    zero_padding=obj.zero_padding,
    pool_size=obj.pool_size,
    strides=obj.strides,
    padding=obj.padding
)
print("Down Sampled Matrix:")
print(down_sampled)
print("Shape after Down Sampling:", down_sampled.shape)





# Test up_sample_population method
upsampled_population = obj.up_sample_population(
    population=population,
    target=t,
    kernel_size=obj.kernel_size,
    padding=obj.up_padding,
    strides=obj.up_strides
)
print("Upsampled Population:")
for idx, up_matrix in enumerate(upsampled_population):
    print(f"Matrix {idx}:")
    print(up_matrix)
    print("Shape after Upsampling:", up_matrix.shape)

