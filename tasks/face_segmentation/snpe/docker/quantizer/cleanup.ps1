# this is a handy powershell script that can cleanup old images from your azure container registry.
# You can find the
<#
.SYNOPSIS
    .
.DESCRIPTION
    .
.PARAMETER password
    Specifies a password that you can find in your Azure portal for the container registry
    under the tab named Access Keys.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$password
)


$registry_name = "snpecontainerregistry001"

$tags = &az acr repository show-tags -n $registry_name --repository quantizer | ConvertFrom-JSON

$latest = [Version]"0"
foreach ($t in $tags) {
    Write-Host "Found tag $t"
    $v = [Version]$t
    if ($v -gt $latest){
        $latest = $v
    }
}

$a = Read-Host "Do you want to delete all images except the latest version $latest (y/n)? "
if ($a -ne "y") {
    Exit 1
}

foreach ($t in $tags) {
    $v = [Version]$t
    if ($v -ne $latest) {
        Write-Host "Deleting image quantizer:$t"
        Write-Host "az acr repository delete --name $registry_name --image quantizer:$v -u $registry_name -p $password"
        $rc = &az acr repository delete --name $registry_name --image quantizer:$v -u $registry_name -p $password --yes
    }
}