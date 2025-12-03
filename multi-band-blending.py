import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_gaussian_pyramid(image, levels):
    """Build Gaussian pyramid for an image."""
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    """Build Laplacian pyramid from Gaussian pyramid."""
    laplacian_pyramid = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # Upsample the next level
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1])
        
        # Match dimensions if needed (pyrUp might create slightly different size)
        if upsampled.shape != gaussian_pyramid[i].shape:
            upsampled = cv2.resize(upsampled, 
                                  (gaussian_pyramid[i].shape[1], 
                                   gaussian_pyramid[i].shape[0]))
        
        # Laplacian = current level - upsampled next level
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    
    # Add the smallest level (top of Gaussian pyramid)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return laplacian_pyramid


def build_mask_pyramid(shape, levels, mask_type='vertical', position=0.5):
    """
    Build mask pyramid for blending.
    
    Args:
        shape: tuple of (height, width)
        levels: number of pyramid levels
        mask_type: 'vertical' or 'horizontal' split
        position: where to split (0.0 to 1.0)
    """
    height, width = shape[:2]
    
    # Create base mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    if mask_type == 'vertical':
        split_point = int(width * position)
        mask[:, :split_point] = 1.0
    else:  # horizontal
        split_point = int(height * position)
        mask[:split_point, :] = 1.0
    
    # Apply Gaussian blur for smooth transition
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    
    # Build pyramid
    mask_pyramid = [mask]
    for i in range(levels - 1):
        mask = cv2.pyrDown(mask)
        mask_pyramid.append(mask)
    
    return mask_pyramid


def blend_pyramids(laplacian1, laplacian2, mask_pyramid):
    """Blend two Laplacian pyramids using mask pyramid."""
    blended_pyramid = []
    
    for lap1, lap2, mask in zip(laplacian1, laplacian2, mask_pyramid):
        # Expand mask to match image channels
        if len(lap1.shape) == 3:
            mask = np.stack([mask] * lap1.shape[2], axis=2)
        
        # Blend: lap1 * mask + lap2 * (1 - mask)
        blended = lap1 * mask + lap2 * (1 - mask)
        blended_pyramid.append(blended)
    
    return blended_pyramid


def reconstruct_from_laplacian(laplacian_pyramid):
    """Reconstruct image from Laplacian pyramid."""
    reconstructed = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Upsample
        reconstructed = cv2.pyrUp(reconstructed)
        
        # Match dimensions if needed
        if reconstructed.shape != laplacian_pyramid[i].shape:
            reconstructed = cv2.resize(reconstructed,
                                      (laplacian_pyramid[i].shape[1],
                                       laplacian_pyramid[i].shape[0]))
        
        # Add Laplacian level
        reconstructed = cv2.add(reconstructed, laplacian_pyramid[i])
    
    return reconstructed


def multiband_blend(image1, image2, levels=6, mask_type='vertical', position=0.5):
    """
    Perform multi-band blending on two images.
    
    Args:
        image1: first input image (numpy array)
        image2: second input image (numpy array)
        levels: number of pyramid levels (default: 6)
        mask_type: 'vertical' or 'horizontal' (default: 'vertical')
        position: blend position 0.0 to 1.0 (default: 0.5)
    
    Returns:
        Blended image (numpy array)
    """
    # Ensure images are the same size
    if image1.shape != image2.shape:
        # Resize to match the smaller dimension
        h = min(image1.shape[0], image2.shape[0])
        w = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (w, h))
        image2 = cv2.resize(image2, (w, h))
    
    # Convert to float for processing
    img1_float = image1.astype(np.float32)
    img2_float = image2.astype(np.float32)
    
    # Build Gaussian pyramids
    gaussian1 = build_gaussian_pyramid(img1_float, levels)
    gaussian2 = build_gaussian_pyramid(img2_float, levels)
    
    # Build Laplacian pyramids
    laplacian1 = build_laplacian_pyramid(gaussian1)
    laplacian2 = build_laplacian_pyramid(gaussian2)
    
    # Build mask pyramid
    mask_pyramid = build_mask_pyramid(image1.shape, levels, mask_type, position)
    
    # Blend pyramids
    blended_pyramid = blend_pyramids(laplacian1, laplacian2, mask_pyramid)
    
    # Reconstruct final image
    result = reconstruct_from_laplacian(blended_pyramid)
    
    # Clip and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


# Example usage
if __name__ == "__main__":
    # Load two images
    img1 = cv2.imread('image1.png')
    img2 = cv2.imread('image2.png')
    
    # Convert BGR to RGB for display
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Perform multi-band blending
    print("Performing multi-band blending...")
    blended = multiband_blend(img1, img2, levels=6, mask_type='vertical', position=0.5)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    # For comparison: simple alpha blending
    print("Performing simple alpha blending for comparison...")
    alpha = 0.5
    simple_blend = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    simple_blend_rgb = cv2.cvtColor(simple_blend, cv2.COLOR_BGR2RGB)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(blended_rgb)
    axes[1, 0].set_title('Multi-Band Blending')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(simple_blend_rgb)
    axes[1, 1].set_title('Simple Alpha Blending (for comparison)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiband_blend_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save the result
    cv2.imwrite('multiband_blended.jpg', blended)
    print("Blending complete! Results saved.")
    
    # Try different blending parameters
    print("\nTrying different blend positions...")
    positions = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, pos in enumerate(positions):
        result = multiband_blend(img1, img2, levels=6, mask_type='vertical', position=pos)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        axes[i].imshow(result_rgb)
        axes[i].set_title(f'Blend Position: {pos}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiband_blend_positions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nDone! Check the output images.")